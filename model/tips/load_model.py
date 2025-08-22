import io
import numpy as np
import torch
import os

from .text_encoder import TextEncoder, Tokenizer
from .image_encoder import vit_small, vit_base, vit_large, vit_so400m, vit_giant2

MAX_LEN = 64
VOCAB_SIZE = 32000
PATCH_SIZE = 14

vision_models = {
        'S': vit_small,
        'B': vit_base,
        'L': vit_large,
        'So400m': vit_so400m,
        'g': vit_giant2,
    }
def load_image_encoder(model_path, model_variant, is_low_res):
    image_size = 224 if is_low_res else 448
    model_def = vision_models[model_variant]
    ffn_layer = 'swiglu' if model_variant == 'g' else 'mlp'

    # Load checkpoint.
    checkpoint = dict(np.load(model_path, allow_pickle=False))
    for key in checkpoint:
        checkpoint[key] = torch.tensor(checkpoint[key])

    with torch.no_grad():
        model = model_def(
            img_size=image_size,
            patch_size=PATCH_SIZE,
            ffn_layer=ffn_layer,
            block_chunks=0,
            init_values=1.0,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        )
        model.load_state_dict(checkpoint)
    return model
  
text_models = {
    'S': {'hidden_size': 384, 'mlp_dim': 1536, 'num_heads': 6, 'num_layers': 12},
    'B': {'hidden_size': 768, 'mlp_dim': 3072, 'num_heads': 12, 'num_layers': 12},
    'L': {'hidden_size': 1024, 'mlp_dim': 4096, 'num_heads': 16, 'num_layers': 12},
    'So400m': {'hidden_size': 1152, 'mlp_dim': 4304, 'num_heads': 16, 'num_layers': 27},
    'g': {'hidden_size': 1536, 'mlp_dim': 6144, 'num_heads': 24, 'num_layers': 12},
}
def load_text_encoder(model_path, model_variant, tokenizer_path):
    with open(model_path, 'rb') as fin:
        inbuffer = io.BytesIO(fin.read())
    np_weights_text = np.load(inbuffer, allow_pickle=False)

    weights_text = {}
    for key, value in np_weights_text.items():
        weights_text[key] = torch.from_numpy(value)
    temperature = weights_text.pop('temperature')

    with torch.no_grad():
        # Define the text model.
        model_text = TextEncoder(
            text_models[model_variant],
            vocab_size=VOCAB_SIZE,
        )
        model_text.load_state_dict(weights_text)

        tokenizer_obj = Tokenizer(tokenizer_path=tokenizer_path)
    return model_text, tokenizer_obj, temperature

model_names = {
    'S': 'tips_oss_s14_{}_distilled_{}.npz',
    'B': 'tips_oss_b14_{}_distilled_{}.npz',
    'L': 'tips_oss_l14_{}_distilled_{}.npz',
    'So400m': 'tips_oss_so400m14_{}_largetext_distilled_{}.npz',
    'g': 'tips_oss_g14_{}_{}.npz',
}
def get_model(path, model_variant, is_low_res):
    res = 'lowres' if is_low_res else 'highres'
    image_encoder_path = os.path.join(path, model_names[model_variant].format(res, 'vision'))
    text_encoder_path = os.path.join(path, model_names[model_variant].format(res, 'text'))
    tokenizer_path = os.path.join(path, 'tokenizer.model')
    
    image_encoder = load_image_encoder(image_encoder_path, model_variant, is_low_res)
    text_encoder, tokenizer, temperature = load_text_encoder(text_encoder_path, model_variant, tokenizer_path)
    
    return image_encoder, text_encoder, tokenizer, temperature