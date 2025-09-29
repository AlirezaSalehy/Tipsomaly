import io
import numpy as np
import torch
import os

from .text_encoder import TextEncoder, Tokenizer
# from .image_encoder import vit_small, vit_base, vit_large, vit_so400m, vit_giant2
from .image_encoder import Block, VisionTransformer, MemEffAttention
import functools

from .checkpoints import checkpoint

MAX_LEN = 64
VOCAB_SIZE = 32000
PATCH_SIZE = 14

vision_models = {
    'S':      {'embed_dim': 384,  'depth': 12, 'num_heads': 6,  'mlp_ratio': 4.0},
    'B':      {'embed_dim': 768,  'depth': 12, 'num_heads': 12, 'mlp_ratio': 4.0},
    'L':      {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4.0},
    'So400m': {'embed_dim': 1152, 'depth': 27, 'num_heads': 16, 'mlp_ratio': 4304/1152},
    'G':      {'embed_dim': 1536, 'depth': 40, 'num_heads': 24, 'mlp_ratio': 4.0},
}
def build_vision_encoder(cfg: dict, *, img_size, patch_size, ffn_layer):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        block_fn=functools.partial(Block, attn_class=MemEffAttention),
        num_register_tokens=1,
        img_size=img_size,
        ffn_layer=ffn_layer,
        block_chunks=0,
        init_values=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    
def load_image_encoder(model_weights_path, model_variant, is_low_res, patch_size = 14):
    img_size = 224 if is_low_res else 448
    ffn_layer = 'swiglu' if model_variant == 'G' else 'mlp'
    cfg = vision_models[model_variant]

    checkpoint_np = dict(np.load(model_weights_path, allow_pickle=False))
    checkpoint = {k: torch.tensor(v) for k, v in checkpoint_np.items()}

    with torch.no_grad():
        model = build_vision_encoder(cfg, img_size=img_size, patch_size=patch_size, ffn_layer=ffn_layer)
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        # Optional: sanity logs
        if missing:
            print(f"[vision:{model_variant}] Missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[vision:{model_variant}] Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    return model
  
text_models = {
    'S': {'hidden_size': 384, 'mlp_dim': 1536, 'num_heads': 6, 'num_layers': 12},
    'B': {'hidden_size': 768, 'mlp_dim': 3072, 'num_heads': 12, 'num_layers': 12},
    'L': {'hidden_size': 1024, 'mlp_dim': 4096, 'num_heads': 16, 'num_layers': 12},
    'So400m': {'hidden_size': 1152, 'mlp_dim': 4304, 'num_heads': 16, 'num_layers': 27},
    'G': {'hidden_size': 1536, 'mlp_dim': 6144, 'num_heads': 24, 'num_layers': 12},
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


CHECKPOINT_TO_VARIANT = {
    "s14h": "S",
    "b14h": "B",
    "l14h": "L",
    "so4h": "So400m",   # 400m = big SoViT model
    "g14l": "G",        # Giant low-res
    "g14h": "G",        # Giant high-res
}
def get_model(model_path, model_checkpoint):
    paths = checkpoint.ensure_model_files(model_checkpoint, model_path)
    for key, path in paths.items():
        print(f"{key}: {path}")
    
    tokenizer_path = os.path.join(model_path, 'tokenizer.model')
    image_enc_name, text_enc_name = checkpoint._model_files_for_basename(checkpoint.MODEL_REGISTRY[model_checkpoint])
    image_encoder_path = os.path.join(model_path, image_enc_name)
    text_encoder_path = os.path.join(model_path, text_enc_name)
    is_low_res = model_checkpoint.endswith('l')
    
    model_variant = CHECKPOINT_TO_VARIANT[model_checkpoint]
    image_encoder = load_image_encoder(image_encoder_path, model_variant, is_low_res)
    text_encoder, tokenizer, temperature = load_text_encoder(text_encoder_path, model_variant, tokenizer_path)
    
    return image_encoder, text_encoder, tokenizer, temperature