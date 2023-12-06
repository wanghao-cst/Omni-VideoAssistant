import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO

import cv2

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    # import pdb;pdb.set_trace()
    model_name = get_model_name_from_path(args.model_path) # 'omni-vicuna-7b-v1.3-finetune_lora_finished'
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else: # True
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs # '<image>\nPlease summarize the key points and main ideas from the following video.'

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

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() # "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nPlease summarize the key points and main ideas from the following video. ASSISTANT:"

    assert args.image_file is not None or args.video_file is not None
    
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
        
    if args.image_file is not None:
        image = load_image(args.image_file)
        if args.image_aspect_ratio == "pad":
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    # import pdb;pdb.set_trace()
    trans_frames = []
    if args.video_file is not None:
        video = cv2.VideoCapture(args.video_file)
            
        fps = video.get(cv2.CAP_PROP_FPS) # 29.9
        frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH) # 480.0
        frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) # 270.0
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT) # 609.0
        duration = frame_count / fps # 20.32

        if frame_count < 700:
            sample_rate = 6
        elif frame_count < 1300:
            sample_rate = 12
        elif frame_count < 2500:
            sample_rate = 24
        else:
            # sample_rate = 48
            sample_rate = max(1,frame_count//50)
        # sample_rate = max(1,frame_count//50)

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
                    rgb_frame = expand2square(pil_image, tuple(int(x*255) for x in image_processor.image_mean))
                        
                video_frames.append(rgb_frame)
            cnt += 1
        video.release()
        for rgb in video_frames:
            trans_frames.append(image_processor.preprocess(rgb, return_tensors='pt')['pixel_values'][0])
        trans_frames_ = torch.stack(trans_frames).to(torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # import pdb;pdb.set_trace()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            videos=[trans_frames_ if trans_frames else image_tensor], # torch.Size([81, 3, 224, 224])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")
    args = parser.parse_args()

    eval_model(args)
