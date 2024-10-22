import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image
from torch import nn
class Virchow(nn.Module):
    def __init__(self, ):
        super().__init__()
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.base_model = model
        self.transforms = transforms

    def forward(self, x):
        output = self.base_model(x)
        class_token = output[:, 0]    # size: B x 1280
        patch_tokens = output[:, 1:]  # size: B x 256 x 1280
        
        # concatenate class token and average pool of patch tokens
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: B x 2560
        return embedding

    def return_transorm(self,):
        return self.transforms



