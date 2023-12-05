import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import math
from tqdm import tqdm
import json
import os
import requests
from PIL import Image
from io import BytesIO

import cv2


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def expand2square(pil_img, background_color):
    # import pdb;pdb.set_trace()
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def get_model_output(model, video_processor, tokenizer, video_file, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else: # True
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs + " Answer in short phrase." # '<image>\nPlease summarize the key points and main ideas from the following video.'

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() # "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nPlease summarize the key points and main ideas from the following video. ASSISTANT:"
    # import pdb;pdb.set_trace()
    trans_frames = []
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS) # 29.9
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH) # 480.0
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) # 270.0
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT) # 609.0
    duration = frame_count / fps # 20.32
    sample_rate = max(1,frame_count//50)

    video_frames = []
    cnt = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if cnt%sample_rate==0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # (270, 480, 3):h w c
        
            if args.image_aspect_ratio == "pad":
                pil_image = Image.fromarray(rgb_frame)
                rgb_frame = expand2square(pil_image, tuple(int(x*255) for x in video_processor.image_mean))
                    
            video_frames.append(rgb_frame)
        cnt += 1
    video.release()
    for rgb in video_frames:
        trans_frames.append(video_processor.preprocess(rgb, return_tensors='pt')['pixel_values'][0])
    trans_frames_ = torch.stack(trans_frames).to(torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # import pdb;pdb.set_trace()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            videos=[trans_frames_], # torch.Size([81, 3, 224, 224])
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs

def eval_model(args):
    # Model
    disable_torch_init()
    # import pdb;pdb.set_trace()
    model_name = get_model_name_from_path(args.model_path) # 'omni-vicuna-7b-v1.3-finetune_lora_finished'
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif 'omni' in model_name.lower(): # True
        conv_mode = "omni_v1"
        # conv_mode = "llava_v1"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else: # True
        args.conv_mode = conv_mode

    gt_qa = json.load(open(args.gt_file_qa, "r"))
    gt_qa = get_chunk(gt_qa, args.num_chunks, args.chunk_idx)

    assert args.output_dir is not None
    answers_file = os.path.join(args.output_dir, f"{model_name}_{args.chunk_idx}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")
    output_list = []
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    index = 0
    for sample in tqdm(gt_qa):
        video_name = "video"+str(sample['video_id'])
        question = sample['question']
        id = sample['id']
        answer = sample['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            # import pdb;pdb.set_trace()
            if os.path.exists(temp_path):
                video_path = temp_path
                # try:
                # Run inference on the video and add the output to the list
                output = get_model_output(model, processor, tokenizer, video_path, question, args)
                sample_set['pred'] = output
                output_list.append(sample_set)
                # except Exception as e:
                #     print(f"Error processing video file '{video_name}': {e}")
                ans_file.write(json.dumps(sample_set) + "\n")
                break

    ans_file.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-dir", type=str, default="/data/wsh/dataset/TrainValVideo/")
    parser.add_argument("--gt_file_qa", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--image-aspect-ratio", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./msrvtt_eval_result/")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
