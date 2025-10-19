from big_vision.models.proj.image_text import two_towers as model_mod

# Remember that the each module like pp should be imported under the same name 
# if some other internal files use big_vision.pp and i use .pp then it run through the 
# module pp twice
from big_vision.pp import builder as pp_builder 
from big_vision.pp import ops_general  
from big_vision.pp import ops_image  
from big_vision.pp import ops_text  
from big_vision.pp.proj.image_text import ops_naflex  
from big_vision.pp.proj.paligemma import ops  
import PIL

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

# images = [PIL.Image.open(fname) for fname in [
#     'apple-ipod.jpg',
#     'apple-blank.jpg',
#     'cold_drink.jpg',
#     'hot_drink.jpg',
#     'caffeine.jpg',
#     'siglip.jpg',
#     'authors.jpg',
#     'robosign.jpg',
#     'cow_beach.jpg',
#     'cow_beach2.jpg',
#     'mountain_view.jpg',
# ]]
# pp_img = pp_builder.get_preprocess_fn(f'resize({RES})|value_range(-1, 1)')
# imgs = np.array([pp_img({'image': np.array(image)})['image'] for image in images])
# print('imgs', imgs.shape)

class InputTransform:
    def __init__(self, RES):
        self.transform = pp_builder.get_preprocess_fn(f'resize({RES})|value_range(-1, 1)')

    def __call__(self, img):
        # print(type(img))
        if np.array(img).size < 3:
            raise ValueError("invalid image: fewer than 3 elements")
        return np.array(self.transform({'image': np.array(img)})['image'])

class TargetTransform:
    def __init__(self, RES):
        self.transform = pp_builder.get_preprocess_fn(f'resize({RES})|value_range(0, 1)')

    def __call__(self, img):
        # print(type(img))
        # print(np.expand_dims(np.array(img), axis=-1).shape)

        out = np.array(self.transform({'image': np.expand_dims(np.array(img), axis=-1)})['image'])
        # print(out.shape)
        return np.squeeze(out, axis=-1)

def create_preprocessors_siglip2(RES):
    # transform = transforms.Compose([
    #     Ensure3Channels(),
    #     transforms.Resize((image_size, image_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    # ])

    # target_transform = transforms.Compose([
    #     transforms.Resize((image_size, image_size)),
    #     transforms.ToTensor(),
    # ])
    pp_input_img = InputTransform(RES)
    pp_target_img = TargetTransform(RES)
    return pp_input_img, pp_target_img

def input_transforms(images, RES):
    pp_img = pp_builder.get_preprocess_fn(f'resize({RES})|value_range(-1, 1)')
    imgs = np.array([pp_img({'image': np.array(image)})['image'] for image in images])
    return imgs 

def target_transforms(images, RES):
    pp_img = pp_builder.get_preprocess_fn(f'resize({RES})|value_range(0, 1)')
    imgs = np.array([pp_img({'image': np.array(image)})['image'] for image in images])
    return imgs

def load(VARIANT, RES, ROOT_PATH='/kaggle/working/cpt/'):
    CKPT = f'siglip2_{VARIANT.lower().replace("/", "")}_{RES}.npz'
    TXTVARIANT, PATCH_SIZE = VARIANT.split('/')
    EMBDIM = {'B': 768, 'L': 1024, 'So400m': 1152, 'g-opt': 1536}[TXTVARIANT]
    # Note: The g-opt vision encoder is paired with a So400m text encoder
    TXTVARIANT = 'So400m' if TXTVARIANT == 'g-opt' else TXTVARIANT
    PATCH_SIZE = int(PATCH_SIZE)
    VOCAB = 256_000
    SEQLEN = 64
    
    # It is significantly faster to first copy the checkpoint (30s vs 8m30 for B and 1m vs ??? for L)
    # !test -f {ROOT_PATH}/{CKPT} || gsutil cp gs://big_vision/siglip2/{CKPT} {ROOT_PATH}
    # print(f'{ROOT_PATH}/{CKPT} ', f'gs://big_vision/siglip2/{CKPT} ')

    model_cfg = ml_collections.ConfigDict(dict(
        image_model='vit',
        image=dict(
            pool_type='map',
            scan=True,
            variant=VARIANT,
        ),
        text_model='proj.image_text.text_transformer',
        text=dict(
            scan=True,
            variant=TXTVARIANT,
            vocab_size=256_000,
        ),
        out_dim=[None, EMBDIM],
        bias_init=-10,  # without this arg, no "b" param is added
    ))
    model = model_mod.Model(**model_cfg)

    # Using `init_params` is slower but will lead to `load` below performing sanity-checks.
    # init_params = jax.jit(model.init, backend="cpu")(jax.random.PRNGKey(42), jnp.zeros([1, RES, RES, 3], jnp.float32), jnp.zeros([1, SEQLEN], jnp.int32))['params']
    init_params = None  # Faster but bypasses loading sanity-checks.
    params = model_mod.load(init_params, f'/{ROOT_PATH}/{CKPT}', model_cfg)

    return model, params


class SigLIPTokenizer:
    def __init__(self, SEQLEN):
        self.pp_txt = pp_builder.get_preprocess_fn(f'lower(key="text")|tok(length={SEQLEN}, model="gemma", bos="no", eos="sticky", key="text")')    

    def __call__(self, texts):
        """texts: str | list[str] -> np.ndarray[int] (B, L)"""
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self.pp_txt({'text': t})['text'] for t in texts])


class SigLIPImageEncoder:
    def __init__(self, model, params):
        self.model = model
        self.params = params

    def __call__(self, imgs):
        return self.model.apply({'params': self.params}, imgs, None)


class SigLIPTextEncoder:
    def __init__(self, model, params):
        self.model = model
        self.params = params

    def __call__(self, text_ids):
        return self.model.apply({'params': self.params}, None, text_ids)

def build_siglip_modules(model_version, image_size, SEQLEN=64):
    model, params = load(model_version, image_size)
    tok = SigLIPTokenizer(SEQLEN)
    img_enc = SigLIPImageEncoder(model, params)
    txt_enc = SigLIPTextEncoder(model, params)
    return img_enc, txt_enc, tok