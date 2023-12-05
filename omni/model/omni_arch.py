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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class OmniMetaModel:

    def __init__(self, config):
        super(OmniMetaModel, self).__init__(config)
        # import pdb;pdb.set_trace()
        if hasattr(config, "mm_vision_tower"): # train False, v1.5 continue finetune True
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            # import pdb;pdb.set_trace()
        if hasattr(config, "mm_video_fuser"): 
            # self.frames_conv = nn.Conv2d(256, 256, kernel_size=(12,1), stride=(10,1)) # b 256 51 4096
            self.frames_conv = nn.Conv2d(576, 576, kernel_size=(12,1), stride=(10,1))
        # self.frames_conv = nn.Conv2d(256, 256, kernel_size=(12,1), stride=(10,1)) # b 256 51 4096 for exp1 test uncomment it 
            

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None): # Train
        vision_tower = model_args.vision_tower # 'openai/clip-vit-large-patch14'
        mm_vision_select_layer = model_args.mm_vision_select_layer      # -2
        mm_vision_select_feature = model_args.mm_vision_select_feature  # patch
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter    # '/home/wanghao/weights/llava/llava-pretrain-vicuna-7b-v1.3/mm_projector.bin'

        self.config.mm_vision_tower = vision_tower
        
        # import pdb;pdb.set_trace()
        # vision_tower = build_vision_tower(model_args)
        if self.get_vision_tower() is None: ## 初次fintune会走这，且require_grad=True，continue时fromepretrain已经有
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else: ## Implement continue finetuning.
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size       # 1024
        self.config.mm_vision_select_layer = mm_vision_select_layer # -2
        self.config.mm_vision_select_feature = mm_vision_select_feature # patch
        
        # self.mm_projector = build_vision_projector(self.config) # 1024->4096
        if getattr(self, 'mm_projector', None) is None: ## 初次fintune会走这，且require_grad=True，continue时fromepretrain已经有
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # import pdb;pdb.set_trace()
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k} # weight:torch.Size([4096, 1024])  bias:torch.Size([4096])
            # import pdb;pdb.set_trace()
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector')) 
            # v1.5: mm_projector_weights['model.mm_projector.0.weight'].shape: torch.Size([4096, 1024])
            # model.mm_projector.0.bias: torch.Size([4096]); model.mm_projector.2.weight: torch.Size([4096, 4096]); model.mm_projector.2.bias: torch.Size([4096])
        if getattr(self, 'frames_conv', None) is None: ## Implement continue finetuning.
            # self.frames_attn = MultiheadAttention(256*4096, num_heads)
            # self.frames_conv = nn.Conv2d(4096, 4096, kernel_size=(12,1), stride=(10,1)) # b 4096 51 256
            
            # self.frames_conv = nn.Conv2d(256, 256, kernel_size=(12,1), stride=(10,1)) # b 256 51 4096
            self.frames_conv = nn.Conv2d(576, 576, kernel_size=(12,1), stride=(10,1)) # b 256 51 4096
            
            # self.keyframes_attn = MultiheadAttention(256*4096, num_heads)
            # import pdb;pdb.set_trace()
        self.config.mm_video_fuser = 'frames_conv'


class OmniMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_frames(self, frames):
        frames_features = self.get_model().get_vision_tower()(frames)  # torch.Size([276, 256, 1024])
        frames_features = self.get_model().mm_projector(frames_features) # torch.Size([276, 256, 4096]) torch.float16
        return frames_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, videos
    ):
        vision_tower = self.get_vision_tower()
        # import pdb;pdb.set_trace()
        # frames_attn = self.get_model().frames_attn
        frames_conv = self.get_model().frames_conv
        # keyframes_attn = self.get_model().keyframes_attn
        # import pdb;pdb.set_trace()
        if vision_tower is None or videos is None or input_ids.shape[1] == 1: # False
            if past_key_values is not None and vision_tower is not None and videos is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        # videos = [torch.Size([51, 3, 224, 224]), torch.Size([79, 3, 224, 224]), torch.Size([60, 3, 224, 224]), torch.Size([86, 3, 224, 224])]
        assert type(videos) is list or videos.ndim == 5 # True
        
        concat_frames = torch.cat([video for video in videos], dim=0) #                       torch.Size([79, 3, 336, 336])
        # import pdb;pdb.set_trace()
        frames_features = self.encode_frames(concat_frames) # torch.Size([276, 256, 4096])    torch.Size([79, 576, 4096])
        split_sizes = [video.shape[0] for video in videos] # [51, 79, 60, 86]
        frames_features = torch.split(frames_features, split_sizes, dim=0) # (torch.Size([51, 256, 4096]), torch.Size([79, 256, 4096]), torch.Size([60, 256, 4096]), torch.Size([86, 256, 4096]))
        # import pdb;pdb.set_trace()
        # frames_features = [x.flatten(0, 1) for x in frames_features]
        key_frames_feature = []
        for frame_feature in frames_features:
            # import pdb;pdb.set_trace()
            frame_feature = frame_feature.unsqueeze(0) # b 51 256 4096
            frame_feature = frame_feature.permute(0,2,1,3) # b 256 51 4096
            # short video
            if frame_feature.shape[2] >= 12:
                frame_feature = frames_conv(frame_feature) # torch.Size([1, 256, 4, 4096])
            frame_feature = frame_feature.squeeze(0).permute(1,0,2) # torch.Size([4, 256, 4096])
            
            # key_frames_feature.append(frame_feature[:6])
            # import pdb;pdb.set_trace()
            num_frames = frame_feature.shape[0]
            key_frames_feature.append(frame_feature[::max(1,num_frames//5)][:6]) # v1.5 576 patch
            

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_video_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids): # torch.Size([4, 375])
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0: # 1 False
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_frames_features = key_frames_feature[cur_video_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                # import pdb;pdb.set_trace()
                # cur_input_embeds = torch.cat([cur_input_embeds_1, cur_frames_features[0:0], cur_input_embeds_2], dim=0)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_frames_features[0], cur_input_embeds_2], dim=0)
                # cur_input_embeds = torch.cat([cur_input_embeds_1, cur_input_embeds_2], dim=0)

                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_video_idx += 1
                # import pdb;pdb.set_trace()
                # never enter it
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0] # (tensor([35], device='cuda:0'),)
            cur_new_input_embeds = []
            if labels is not None: # torch.Size([4, 375])
                cur_labels = labels[batch_idx] # torch.Size([375]): -100...labels...-100
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0: # 统计元素个数 1
                cur_frames_features = key_frames_feature[cur_video_idx] # torch.Size([4, 256, 4096])
                cur_frames_features = cur_frames_features.reshape(-1,4096) # torch.Size([1024, 4096])
                
                image_token_start = image_token_indices[0] # tensor(35, device='cuda:0')
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False): # False
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_frames_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_frames_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else: # True
                    # import pdb;pdb.set_trace()
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start])) # instru部分的embed: torch.Size([35, 4096])
                    cur_new_input_embeds.append(cur_frames_features) # torch.Size([1024, 4096]) input加入frames特征
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start]) # torch.Size([35]) 全-100
                        cur_new_labels.append(torch.full((cur_frames_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)) # torch.Size([1024])
                        cur_labels = cur_labels[image_token_start+1:] # 339 = 375-35-1(img_token) 稍后加到cur_new_labels中
                cur_video_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False): # False
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:] # torch.Size([339])
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0] # 空
                
            if cur_input_ids.numel() > 0: # True
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False): # False
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids)) # [torch.Size([35, 4096])固定template,torch.Size([1024, 4096])图像特征,  QA：torch.Size([339, 4096])]
                if labels is not None:
                    cur_new_labels.append(cur_labels) # [torch.Size([35]),torch.Size([1024]),   前面全为-100 torch.Size([339])]
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0) # torch.Size([1398, 4096]): 35+1024+339
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0) # torch.Size([1398])
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds): # True
            max_len = max(x.shape[0] for x in new_input_embeds) # 1910

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else: # False img模式默认只有256长度相等
            # import pdb;pdb.set_trace()
            new_input_embeds = torch.stack(new_input_embeds, dim=0) # torch.Size([4, 716, 4096])   716=461-1imgtoken+256imgfeature
            if labels is not None: # torch.Size([4, 461])
                new_labels  = torch.stack(new_labels, dim=0) # torch.Size([4, 716])
            
            if attention_mask is not None: # torch.Size([4, 461])
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device) # torch.Size([4, 255]个True 相当于256个img特征-1个imgtoken
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1) # torch.Size([4, 716]) 716=461+255(新加入的img特征255个token mask为True)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token: # False
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end: # False
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token: # False
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
