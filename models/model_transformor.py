import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
import math

class Attn(nn.Module):
    def __init__(self, head_dim, use_flash_attention):
        super(Attn, self).__init__()
        self.use_flash_attention = use_flash_attention
        self.head_dim = head_dim

    def scaled_dot_product_attention(self,query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    
    def forward(self, q, k, v, ):
        if self.use_flash_attention:
            
            attn_output = self.scaled_dot_product_attention(q, k, v, is_causal=False)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            attn_scores = F.softmax(attn_scores, dim=-1)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            attn_scores = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_scores, v)

        return attn_output, attn_scores
class MultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self,
        gate = True,
        d_model=1024,
        num_heads=8,
        use_flash_attention=True,
        top_k=1,
        instance_loss_fn=nn.CrossEntropyLoss(), 
        dropout = False, k_sample=8, n_classes=2, subtyping=False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_classes =n_classes
        self.use_flash_attention = use_flash_attention

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        head_dim = d_model // num_heads
        self.attn = Attn(head_dim, use_flash_attention)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Linear(d_model, n_classes)
        def dummy_mlp(size = [d_model, 512, 2]):
            fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
            fc.append(nn.Linear(size[1], size[2]))
            return nn.Sequential(*fc)
        instance_classifiers = [dummy_mlp() for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.instance_loss_fn = instance_loss_fn
        initialize_weights(self)
        self.top_k=top_k
        self.k_sample = k_sample

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for param in self.parameters():
            if isinstance(param, nn.Parameter):
                param.data = param.data.to(device)
    
    def inst_eval(self, A, h, classifier): 
        device = A.device
        h = torch.squeeze(h)
        A = torch.squeeze(A)
        #raise NotImplementedError("Stop")
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    def forward(self, x, label=None, instance_eval=False,):
        
        patch_num, d_model = x.shape
        batch_size = 1# more like a place holder here
        
        x = x.view(batch_size, patch_num, d_model)
        #pos_encoding = self.get_positional_encoding().to(x.device)
        #x = x #+ pos_encoding.view(patch_num, d_model)
        class_tokens = self.class_token.expand(batch_size, -1, -1).to(x.device)
        x = torch.cat([class_tokens, x], dim=1)

        q = (
            self.q_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        

        attn_output, attention_scores = self.attn(q, k, v)
        # Calculate instance attention scores
        A = attention_scores[:, :, 0, 1:].mean(dim=1)  # Average over heads, exclude class token

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.out_proj(attn_output)

        class_token_output = output[:, 0]
        h = output[:, 1:]

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    continue
                total_inst_loss += instance_loss

        logits = self.classifier(class_token_output)
        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.num_classes).view(-1, 1), (m % self.num_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]
        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}

        return top_instance, Y_prob, Y_hat, y_probs, results_dict
    


class MultiHeadAttention_CL(nn.Module):
    def __init__(
        self,
        d_model=1024,
        num_heads=8,
        n_classes=2,
        use_flash_attention=True,
        top_k=1,
        instance_loss_fn=nn.CrossEntropyLoss(), 
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_classes =n_classes
        self.use_flash_attention = use_flash_attention

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        head_dim = d_model // num_heads
        self.attn = Attn(head_dim, use_flash_attention)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Linear(d_model, n_classes)

        instance_classifiers = [nn.Linear(d_model, 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.instance_loss_fn = instance_loss_fn
        initialize_weights(self)
        self.top_k=top_k
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for param in self.parameters():
            if isinstance(param, nn.Parameter):
                param.data = param.data.to(device)
        
        
    
    def forward(self, x):
        
        patch_num, d_model = x.shape
        batch_size = 1# more like a place holder here

        x = x.view(batch_size, patch_num, d_model)
        #pos_encoding = self.get_positional_encoding().to(x.device)
        x = x #+ pos_encoding.view(patch_num, d_model)

        class_tokens = self.class_token.expand(batch_size, -1, -1).to(x.device)
        x = torch.cat([class_tokens, x], dim=1)

        q = (
            self.q_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_output = self.attn(q, k, v)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.out_proj(attn_output)

        class_token_output = output[:, 0]
        logits = self.classifier(class_token_output)
        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.num_classes).view(-1, 1), (m % self.num_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]
        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}
        return top_instance, Y_prob, Y_hat, y_probs, results_dict
    
    def get_positional_encoding(self):
        raise NotImplementedError("I need to think about how to invole this now")
        pos_encoding = torch.zeros(self.patch_num, self.d_model)
        assert pos_encoding.shape == (
            self.patch_num,
            self.d_model,
        ), f"pos_encoding shape mismatch: {pos_encoding.shape}"

        # Generate a range of positions for height and width
        y_pos = (
            torch.arange(height).unsqueeze(1).float()
        )  # Unsqueeze to make it a column vector
        assert y_pos.shape == (height, 1), f"y_pos shape mismatch: {y_pos.shape}"

        x_pos = (
            torch.arange(width).unsqueeze(1).float()
        )  # Unsqueeze to make it a row vector
        assert x_pos.shape == (width, 1), f"x_pos shape mismatch: {x_pos.shape}"

        # Calculate the divisor term for the positional encoding formula
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * -(math.log(10000.0) / self.d_model)
        )
        assert div_term.shape == (
            self.d_model // 2,
        ), f"div_term shape mismatch: {div_term.shape}"

        # Apply the sine function to the y positions and expand to match (height, 1, d_model // 2)
        pos_encoding_y_sin = (
            torch.sin(y_pos * div_term)
            .unsqueeze(1)
            .expand(height, width, self.d_model // 2)
        )
        assert pos_encoding_y_sin.shape == (
            height,
            width,
            self.d_model // 2,
        ), f"pos_encoding_y_sin shape mismatch: {pos_encoding_y_sin.shape}"

        # Apply the cosine function to the y positions and expand to match (height, 1, d_model // 2)
        pos_encoding_y_cos = (
            torch.cos(y_pos * div_term)
            .unsqueeze(1)
            .expand(height, width, self.d_model // 2)
        )
        assert pos_encoding_y_cos.shape == (
            height,
            width,
            self.d_model // 2,
        ), f"pos_encoding_y_cos shape mismatch: {pos_encoding_y_cos.shape}"

        # Apply the sine function to the x positions and expand to match (1, width, d_model // 2)

        pos_encoding_x_sin = (
            torch.sin(x_pos * div_term)
            .unsqueeze(0)
            .expand(height, width, self.d_model // 2)
        )
        assert pos_encoding_x_sin.shape == (
            height,
            width,
            self.d_model // 2,
        ), f"pos_encoding_x_sin shape mismatch: {pos_encoding_x_sin.shape}"

        # Apply the cosine function to the x positions and expand to match (1, width, d_model // 2)
        pos_encoding_x_cos = (
            torch.cos(x_pos * div_term)
            .unsqueeze(0)
            .expand(height, width, self.d_model // 2)
        )
        assert pos_encoding_x_cos.shape == (
            height,
            width,
            self.d_model // 2,
        ), f"pos_encoding_x_cos shape mismatch: {pos_encoding_x_cos.shape}"

        # Combine the positional encodings
        pos_encoding[:, :, 0::2] = pos_encoding_y_sin + pos_encoding_x_sin
        pos_encoding[:, :, 1::2] = pos_encoding_y_cos + pos_encoding_x_cos

        assert pos_encoding.shape == (
            height,
            width,
            self.d_model,
        ), f"pos_encoding shape mismatch: {pos_encoding.shape}"

        return pos_encoding