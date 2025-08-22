import torch
import torch.nn as nn
from torch.nn import functional as F

# NOTE: Since tips is inside model/, your import in model/omaly/text_encoder.py should be:
from model.tips.text_encoder import TextEncoder as BaseTextEncoder

class text_encoder(nn.Module):
    def __init__(self, tokenizer, tips_text_encoder, MAX_LEN, prompt_learn_method='none'):
        super(text_encoder, self).__init__()  
        self.tokenizer = tokenizer
        self.tips_encoder = tips_text_encoder
        self.MAX_LEN = MAX_LEN
        self.prompt_learn_method = prompt_learn_method

        self.prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw',
                              '{} without defect',
                              '{} without damage']
        self.prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        self.prompt_templates = ['a bad photo of a {}.',
                                 'a low resolution photo of the {}.',
                                 'a bad photo of the {}.',
                                 'a cropped photo of the {}.',
                                 ]
        text_emd_dim = 1024
        num_tokens = 8
    
        if not self.prompt_learn_method == 'none':
            self.normal_prompt = torch.nn.Parameter(torch.randn(num_tokens, text_emd_dim) * 0.02)  # Learnable prompt for normal text description with std of 0.02
            self.abnormal_prompt = torch.nn.Parameter(torch.randn(num_tokens, text_emd_dim) * 0.02)  # Learnable prompt for abnormal text description with std of 0.02
            self.learnable_parameters = torch.nn.ParameterList([self.normal_prompt, self.abnormal_prompt])
            
    def forward(self, texts, device):
        text_feature_list = []

        for indx, text in enumerate(texts):
            text_features = self.encode_text(text, device)
            text_feature_list.append(text_features)

        text_features = torch.stack(text_feature_list, dim=0)
        text_features = F.normalize(text_features, dim=2)
        return text_features

    def encode_text(self, text, device):
        text_features = []
        for i in range(len(self.prompt_state)):
            learnables = self.learnable_parameters[i] if not self.prompt_learn_method == 'none' else None
            
            text = text.replace('-', ' ')
            prompted_state = [state.format(text) for state in self.prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in self.prompt_templates:
                    prompted_sentence.append(template.format(s))
            
            text_ids, text_paddings = self.tokenizer.tokenize(prompted_sentence, max_len=self.MAX_LEN)
            class_embeddings = self.tips_encoder(text_ids.to(device), text_paddings.to(device), learnables, self.prompt_learn_method, device)

            # NOTE: Avoid in-place /=, +=, _add() on tensor which are actively gradiented 
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=0)
        return text_features
