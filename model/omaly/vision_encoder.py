import torch
import torch.nn as nn
from torch.nn import functional as F
import jax.numpy as jnp
import numpy as np

def jax_to_torch(x):
    return torch.from_numpy(np.array(x))
    
# The backbone can be SigLIP2 or TIPS
class vision_encoder(nn.Module):
    def __init__(self, bb_vision_encoder, bb_type):
        super(vision_encoder, self).__init__()
        self._encoder = bb_vision_encoder
        self.model = bb_type

    def _forward_tips(_encoder, images):
        outputs = _encoder(images)
        
        first_cls_token = outputs[0]
        first_cls_token = first_cls_token / first_cls_token.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        
        second_cls_token = outputs[1]
        second_cls_token = second_cls_token / second_cls_token.norm(dim=-1, keepdim=True).clamp(min=1e-3)

        spatial_tokens = outputs[2]
        spatial_tokens = spatial_tokens / spatial_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        
        return first_cls_token, second_cls_token, spatial_tokens
    
    def _forward_siglip2(_encoder, images):
        _, _, out = _encoder(images)
        return jax_to_torch(out['img/normalized']), jax_to_torch(out['img/normalized']), jax_to_torch(out['img/2d_normalized'])
    
    def _forward_siglip2_hf(_encoder, images):
        image_features = _encoder(images)
        return image_features.pooler_output, image_features.pooler_output, image_features.last_hidden_state

    def forward(self, images):
        if self.model == 'siglip2':
            return vision_encoder._forward_siglip2(self._encoder, images)
        elif self.model == 'tips':
            return vision_encoder._forward_tips(self._encoder, images)
        elif self.model == "siglip2-hf":
            return vision_encoder._forward_siglip2_hf(self._encoder, images)
         
    # outputs /= outputs.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    # first_cls_token, second_cls_token, spatial_tokens = outputs[:, 0], outputs[:, 1], outputs[:, 2:]