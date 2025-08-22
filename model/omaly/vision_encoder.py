import torch
import torch.nn as nn
from torch.nn import functional as F

class vision_encoder(nn.Module):
    def __init__(self, tips_vision_encoder):
        super(vision_encoder, self).__init__()
        self.tips_encoder = tips_vision_encoder

    def forward(self, images):
        outputs = self.tips_encoder(images)
        
        first_cls_token = outputs[0]
        first_cls_token = first_cls_token / first_cls_token.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        
        second_cls_token = outputs[1]
        second_cls_token = second_cls_token / second_cls_token.norm(dim=-1, keepdim=True).clamp(min=1e-3)

        spatial_tokens = outputs[2]
        spatial_tokens = spatial_tokens / spatial_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        
        return first_cls_token, second_cls_token, spatial_tokens
    
    # outputs /= outputs.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    # first_cls_token, second_cls_token, spatial_tokens = outputs[:, 0], outputs[:, 1], outputs[:, 2:]