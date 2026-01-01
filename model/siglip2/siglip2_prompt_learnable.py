from typing import Optional
import torch
import torch.nn as nn
from transformers.models.siglip.modeling_siglip import SiglipTextConfig, SiglipTextEmbeddings, SiglipTextTransformer, SiglipTextModel, SiglipPreTrainedModel
from transformers.utils import (
    auto_docstring,
    TransformersKwargs,
    can_return_tuple,
)

from transformers.utils.generic import check_model_inputs
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

class SiglipTextEmbeddingsWithPromptLearning(SiglipTextEmbeddings):
    """
    Extends SiglipTextEmbeddings with learnable prompt tokens.
    Prompts are added BEFORE the position embeddings are applied (so they
    receive position embeddings as part of the combined sequence).
    """
    def __init__(self, config: SiglipTextConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        learnable_prompts: torch.Tensor = None,
        learning_method: str = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids) # [B, L, D]

        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        if learnable_prompts is not None:
            if learning_method == 'concat':
                prompts = learnable_prompts.unsqueeze(0).expand(batch_size, -1, -1).to(device) # [B, P, D]
                inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1) # [B, P+L, D]
            elif learning_method == 'sumate':
                prompt_len = learnable_prompts.size(0)
                inputs_embeds[:, :prompt_len, :] += learnable_prompts.unsqueeze(0)
            elif learning_method == 'entire_learnable':
                inputs_embeds = learnable_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        
        current_len = inputs_embeds.size(1)
        if current_len > seq_length:
            inputs_embeds = inputs_embeds[:, :seq_length, :]
        elif current_len < seq_length: # must check this case if happens or not
            pad_len = seq_length - current_len
            pad_embed = torch.zeros((batch_size, pad_len, inputs_embeds.size(2)))
            inputs_embeds = torch.cat([inputs_embeds, pad_embed], dim=1)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings
    

class SiglipTextTransformerWithPromptLearning(SiglipTextTransformer):
    def __init__(self, config: SiglipTextConfig):
        super().__init__(config)
        self.embeddings = SiglipTextEmbeddingsWithPromptLearning(config)
    
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        learnable_prompts: torch.Tensor = None,
        learning_method: str = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                        learnable_prompts=learnable_prompts, learning_method=learning_method)

        # note: SigLIP's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        uses_flash_attention = "flash" in self.config._attn_implementation
        if uses_flash_attention:
            attention_mask = None
        elif attention_mask is not None and not uses_flash_attention:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # The model uses the last token's hidden state, which may be padding.
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class SiglipTextModelWithPromptLearning(SiglipPreTrainedModel):
    config: SiglipTextConfig
    input_modalities = ("text",)

    def __init__(self, config: SiglipTextConfig):
        super().__init__(config)
        self.text_model = SiglipTextTransformerWithPromptLearning(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        learnable_prompts: torch.Tensor = None,
        learning_method: str = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, SiglipTextModel

        >>> model = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        >>> # important: make sure to set padding="max_length" as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            learnable_prompts=learnable_prompts,
            learning_method=learning_method,
            **kwargs,
        )