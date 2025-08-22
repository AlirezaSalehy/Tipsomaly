import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

class SigLIP2TextEncoderWrapper(nn.Module):
    def __init__(self, model_ckpt, MAX_LEN, prompt_learn_method='none'):
        super(SigLIP2TextEncoderWrapper, self).__init__()
        
        self.model = AutoModel.from_pretrained(model_ckpt)  # Load SigLIP 2 model
        self.processor = AutoProcessor.from_pretrained(model_ckpt)  # To preprocess data for SigLIP 2
        self.MAX_LEN = MAX_LEN
        self.prompt_learn_method = prompt_learn_method

        text_emd_dim = 1024  # Adjust this as per SigLIP 2's config
        num_tokens = 8  # Number of learnable prompt tokens
        if self.prompt_learn_method != 'none':
            self.normal_prompt = torch.nn.Parameter(torch.randn(num_tokens, text_emd_dim) * 0.02)
            self.abnormal_prompt = torch.nn.Parameter(torch.randn(num_tokens, text_emd_dim) * 0.02)
            self.learnable_parameters = torch.nn.ParameterList([self.normal_prompt, self.abnormal_prompt])

    def __call__(self, ids: torch.Tensor, paddings: torch.Tensor, learnable_prompts: torch.Tensor = None, 
                 learning_method: str = None, device='cuda'):
        batch_size, original_seq_length = ids.shape

        text_embeds = self.model.get_input_embeddings()(ids)  # [B, L, D]
        text_embeds = text_embeds * (self.model.config.hidden_size ** 0.5)  # Standard scaling

        text_embeds = text_embeds.to(device)
        
        if learnable_prompts is not None:
            if learning_method == 'concat':
                prompts = learnable_prompts.unsqueeze(0).expand(batch_size, -1, -1).to(text_embeds.device)
                text_embeds = torch.cat([prompts, text_embeds], dim=1)  # [B, P+L, D]
                paddings = torch.cat([torch.zeros((batch_size, prompts.size(1)), device=text_embeds.device), paddings], dim=1)

            elif learning_method == 'sumate':
                prompt_len = learnable_prompts.size(0)
                text_embeds[:, :prompt_len, :] += learnable_prompts.unsqueeze(0)

            elif learning_method == 'entire_learnable':
                text_embeds = learnable_prompts.unsqueeze(0).expand(batch_size, -1, -1)
                paddings = torch.zeros((batch_size, text_embeds.size(1)), device=text_embeds.device)

        # Step 4: Ensure fixed sequence length (truncate or pad)
        current_len = text_embeds.size(1)
        if current_len > original_seq_length:
            text_embeds = text_embeds[:, :original_seq_length, :]
            paddings = paddings[:, :original_seq_length]
        elif current_len < original_seq_length:
            pad_len = original_seq_length - current_len
            pad_embed = torch.zeros((batch_size, pad_len, text_embeds.size(2)), device=text_embeds.device)
            pad_mask = torch.ones((batch_size, pad_len), device=text_embeds.device)
            text_embeds = torch.cat([text_embeds, pad_embed], dim=1)
            paddings = torch.cat([paddings, pad_mask], dim=1)

        # Step 5: Add positional encoding and run through the transformer
        pos_embeddings = self.model.embeddings.position_embeddings(torch.arange(0, original_seq_length, device=device))
        text_embeds = text_embeds + pos_embeddings  # [B, L, D]
        
        # Create attention mask from paddings (1 = attention, 0 = no attention)
        attn_mask = (paddings == 0).float().to(device)  # [B, L]
        
        # Step 6: Run through SigLIP 2's transformer
        transformer_output = self.model.transformer(text_embeds, attention_mask=attn_mask)
        
        # Step 7: Apply final layer norm and pooling (if required)
        pooled_output = transformer_output.last_hidden_state  # [B, L, D]
        pooled_output = pooled_output.mean(dim=1)  # Optionally, pooling

        return pooled_output


class SigLIP2TextEncoderWithPrompts(nn.Module):
    def __init__(self, tokenizer, model_ckpt, MAX_LEN, prompt_learn_method='none'):
        super(SigLIP2TextEncoderWithPrompts, self).__init__()
        self.tokenizer = tokenizer
        self.model_ckpt = model_ckpt
        self.MAX_LEN = MAX_LEN
        self.prompt_learn_method = prompt_learn_method
        
        self.siglip_encoder = SigLIP2TextEncoderWrapper(model_ckpt, MAX_LEN, prompt_learn_method)

    def forward(self, texts, device):
        text_feature_list = []

        for indx, text in enumerate(texts):
            text_features = self.encode_text(text, device)
            text_feature_list.append(text_features)

        text_features = torch.stack(text_feature_list, dim=0)
        text_features = F.normalize(text_features, dim=2)
        return text_features

    def encode_text(self, text, device):
        encoding = self.tokenizer(text, padding=True, truncation=True, max_length=self.MAX_LEN, return_tensors='pt')
        ids = encoding['input_ids'].to(device)
        paddings = encoding['attention_mask'].to(device)

        learnable_prompts = self.siglip_encoder.normal_prompt if self.prompt_learn_method != 'none' else None
        
        class_embeddings = self.siglip_encoder(ids, paddings, learnable_prompts, self.prompt_learn_method, device)
        return class_embeddings
