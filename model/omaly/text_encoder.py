import torch
import torch.nn as nn
from torch.nn import functional as F

# NOTE: Since tips is inside model/, your import in model/omaly/text_encoder.py should be:
from model.tips.text_encoder import TextEncoder as BaseTextEncoder
from .fixed_prompts import generate_prompt_templates
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import math

def jax_to_torch(x):
    return torch.from_numpy(np.array(x))

class text_encoder(nn.Module):
    def __init__(self, tokenizer, bb_text_encoder, bb_type, MAX_LEN, prompt_learn_method='none', prompt_type='industrial', n_prompt=8, n_deep=0, d_deep=0):
        super(text_encoder, self).__init__()  
        self.tokenizer = tokenizer
        self._encoder = bb_text_encoder
        self.model = bb_type
        self.MAX_LEN = MAX_LEN
        self.prompt_learn_method = prompt_learn_method
        self.n_deep_tokens = n_deep
        self.d_deep_tokens = d_deep
        self.n_prompt = n_prompt
        self.prompt_type = prompt_type
        
        self.prompt_normal, self.prompt_abnormal, self.prompt_templates = generate_prompt_templates(self.prompt_type)

        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        text_emd_dim = bb_text_encoder.transformer.width if self.model == 'tips' else bb_text_encoder.model.out_dim[1]
    
        if self.n_deep_tokens > 0 and self.d_deep_tokens > 0:
            self.deep_parameters = torch.nn.ParameterList([torch.nn.Parameter(\
                                    torch.randn(self.n_deep_tokens, text_emd_dim) * 0.02) \
                                    for _ in range(self.d_deep_tokens)])
        else:
            self.deep_parameters = None
    
        if not self.prompt_learn_method == 'none':
            self.normal_prompt = torch.nn.Parameter(torch.randn(self.n_prompt, text_emd_dim) * 0.02)  # Learnable prompt for normal text description with std of 0.02
            self.abnormal_prompt = torch.nn.Parameter(torch.randn(self.n_prompt, text_emd_dim) * 0.02)  # Learnable prompt for abnormal text description with std of 0.02
            self.learnable_prompts = torch.nn.ParameterList([self.normal_prompt, self.abnormal_prompt])
        
        
    def forward(self, texts, device, learned=False):
        text_feature_list = []

        for indx, text in enumerate(texts):
            text_features = self.encode_text(text, device, learned)
            text_feature_list.append(text_features)

        text_features = torch.stack(text_feature_list, dim=0)
        text_features = F.normalize(text_features, dim=2)
        return text_features

    def encode_text(self, text, device, learned=False):
        text_features = []
        for i in range(len(self.prompt_state)):
            learnables = self.learnable_prompts[i] if learned and not self.prompt_learn_method == 'none' else None
            deep_parameters = self.deep_parameters if learned else None
            
            text = text.replace('-', ' ')
            prompted_state = [state.format(text) for state in self.prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in self.prompt_templates:
                    prompted_sentence.append(template.format(s))
            
            if self.model == 'tips':
                # NOTE: replace the class based prompt learning concatenated to the templates with only 2 sentences
                text_ids, text_paddings = self.tokenizer.tokenize(prompted_sentence, max_len=self.MAX_LEN)
                class_embeddings = self._encoder(text_ids.to(device), text_paddings.to(device), learnables, self.prompt_learn_method, deep_parameters, device)

                # NOTE: Avoid in-place /=, +=, _add() on tensor which are actively gradiented 
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-3)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
            
            elif self.model == 'siglip2':
                # We'll accumulate sums in JAX so final mean+normalize happens exactly like original.
                num_prompts = len(prompted_sentence)
                if num_prompts == 0:
                    class_embedding = torch.zeros(getattr(self, "expected_text_dim", 512), device=device)
                    text_features.append(class_embedding)
                    continue

                batch_size = 20
                batch_ranges = range(0, num_prompts, batch_size)
                total_batches = math.ceil(num_prompts / batch_size)

                # accumulator for sum in JAX (None -> initialize on first batch)
                sum_jax = None

                # optional tqdm over batches
                batch_iter = tqdm(batch_ranges, total=total_batches,
                                desc=f"Encoding text {i+1}/{len(self.prompt_state)} ({self.model})",
                                leave=False)

                for start in batch_iter:
                    batch_sentences = prompted_sentence[start:start + batch_size]
                    # tokenizer expects a list of strings (batch)
                    txts = self.tokenizer(batch_sentences)

                    # encoder returns jax arrays; ztxt shape -> (batch, dim)
                    _, ztxt, out = self._encoder(txts)

                    # sum along batch axis in JAX to accumulate raw (unnormalized) vectors
                    batch_sum = jnp.sum(ztxt, axis=0)  # shape (dim,)

                    if sum_jax is None:
                        sum_jax = batch_sum
                    else:
                        sum_jax = sum_jax + batch_sum

                # now compute mean in JAX (exactly as original did: mean then normalize)
                mean_jax = sum_jax / float(num_prompts)
                mean_jax = mean_jax / (jnp.linalg.norm(mean_jax, axis=-1, keepdims=True) + 1e-8)

                # convert final normalized mean to torch once
                class_embedding = jax_to_torch(mean_jax)  # shape: (dim,)

            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=0)
        return text_features