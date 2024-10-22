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

    
    def forward(self, q, k, v, ):
        if self.use_flash_attention:
            attn_output = F.scale_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)
        return attn_output
class MultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self,
        d_model=2048,
        num_heads=8,
        n_classes=2,
        height_max=445,
        width_max=230,
        use_flash_attention=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_classes =n_classes
        self.height_max = height_max
        self.width_max = width_max
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
    
    def forward(self, x):
        batch_size, d_model, height, width = x.shape

        x = x.permute(0, 2, 3, 1).view(batch_size, height * width, d_model)
        pos_encoding = self.get_positional_encoding().to(x.device)
        x = x + pos_encoding.view(height * width, d_model)

        class_tokens = self.class_token.expand(batch_size, -1, -1)
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
        return logits


    def get_positional_encoding(self):
        # Initialize a positional encoding tensor with zeros, shape (height, width, d_model)
        height = self.height_max
        width = self.width_max
        pos_encoding = torch.zeros(height, width, self.d_model)
        assert pos_encoding.shape == (
            height,
            width,
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


