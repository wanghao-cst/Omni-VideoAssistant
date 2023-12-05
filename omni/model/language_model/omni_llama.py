#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..omni_arch import OmniMetaModel, OmniMetaForCausalLM


class OmniConfig(LlamaConfig):
    model_type = "omni"


class OmniLlamaModel(OmniMetaModel, LlamaModel):
    config_class = OmniConfig

    def __init__(self, config: LlamaConfig):
        # import pdb;pdb.set_trace()
        super(OmniLlamaModel, self).__init__(config)


class OmniLlamaForCausalLM(LlamaForCausalLM, OmniMetaForCausalLM): # OmniMetaForCausalLM负责准备多模态label
    config_class = OmniConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        # import pdb;pdb.set_trace()
        self.model = OmniLlamaModel(config) # vision tower+llama
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 4096->32000

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,                         # torch.Size([4, 461])
        attention_mask: Optional[torch.Tensor] = None,              # torch.Size([4, 461])
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # None
        inputs_embeds: Optional[torch.FloatTensor] = None,          # None
        labels: Optional[torch.LongTensor] = None,                  # torch.Size([4, 461])   
        use_cache: Optional[bool] = None,                           # None
        output_attentions: Optional[bool] = None,                   # None
        output_hidden_states: Optional[bool] = None,                # None
        videos: Optional[torch.FloatTensor] = None,                 # torch.Size([4, 3, 224, 224])
        return_dict: Optional[bool] = None,                         # None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # import pdb;pdb.set_trace()
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, videos)
        # input_ids: None
        # attention_mask: torch.Size([4, 1910])
        # past_key_values: None
        # inputs_embeds: torch.Size([4, 1910, 4096])
        # labels: torch.Size([4, 1910])
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        ) # outputs['last_hidden_state']: torch.Size([4, 1910, 4096])

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # torch.Size([4, 1909, 32000]) eos预测的不要了
            shift_labels = labels[..., 1:].contiguous() # torch.Size([4, 1909]) 从bos下一个开始预测
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size) # torch.Size([7636, 32000])
            shift_labels = shift_labels.view(-1) # torch.Size([7636])
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels) # tensor(12.8750, device='cuda:0', dtype=torch.bfloat16,grad_fn=<NllLossBackward0>)
        # import pdb;pdb.set_trace()
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        # import pdb;pdb.set_trace()
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "videos": kwargs.get("videos", None), # vid:[torch.Size([81, 3, 224, 224])] image:[torch.Size([1, 3, 224, 224])]
            }
        )
        return model_inputs

AutoConfig.register("omni", OmniConfig)
AutoModelForCausalLM.register(OmniConfig, OmniLlamaForCausalLM)
