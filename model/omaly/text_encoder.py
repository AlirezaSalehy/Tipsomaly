import torch
import torch.nn as nn
from torch.nn import functional as F

# NOTE: Since tips is inside model/, your import in model/omaly/text_encoder.py should be:
from model.tips.text_encoder import TextEncoder as BaseTextEncoder
from .fixed_prompts import generate_prompt_templates

class text_encoder(nn.Module):
    def __init__(self, tokenizer, tips_text_encoder, MAX_LEN, prompt_learn_method='none', prompt_type='industrial', n_prompt=8, n_deep=0, d_deep=0):
        super(text_encoder, self).__init__()  
        self.tokenizer = tokenizer
        self.tips_encoder = tips_text_encoder
        self.MAX_LEN = MAX_LEN
        self.prompt_learn_method = prompt_learn_method
        self.n_deep_tokens = n_deep
        self.d_deep_tokens = d_deep
        self.n_prompt = n_prompt
        self.prompt_type = prompt_type
        
        self.prompt_normal, self.prompt_abnormal, self.prompt_templates = generate_prompt_templates(self.prompt_type)

        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        text_emd_dim = tips_text_encoder.transformer.width # removing this hard code 1024
    
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
            
            # NOTE: replace the class based prompt learning concatenated to the templates with only 2 sentences
            text_ids, text_paddings = self.tokenizer.tokenize(prompted_sentence, max_len=self.MAX_LEN)
            class_embeddings = self.tips_encoder(text_ids.to(device), text_paddings.to(device), learnables, self.prompt_learn_method, deep_parameters, device)

            # NOTE: Avoid in-place /=, +=, _add() on tensor which are actively gradiented 
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=0)
        return text_features
