import shutil
import subprocess

import torch
import gradio as gr
from fastapi import FastAPI
import os
from PIL import Image
import tempfile
from decord import VideoReader, cpu
from transformers import TextStreamer
import cv2

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle, Conversation
from llava.serve.gradio_utils import Chat, tos_markdown, learn_more_markdown, title_markdown, block_css


def save_image_to_local(image):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image = Image.open(image)
    image.save(filename)
    # print(filename)
    return filename


def save_video_to_local(video_path):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')
    shutil.copyfile(video_path, filename)
    return filename

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

def generate(image, video, textbox_in, first_run, state, state_, images_tensor):
    flag = 1
    if not textbox_in:
        if len(state_.messages) > 0:
            textbox_in = state_.messages[-1][1]
            state_.messages.pop(-1)
            flag = 0
        else:
            return "Please enter instruction"

    image1 = image if image else "none"
    video1 = video if video else "none"

    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()

    first_run = False if len(state.messages) > 0 else True

    vis_processor = handler.processor
    if os.path.exists(image1) and not os.path.exists(video1):
        image = Image.open(image1).convert('RGB')
        image = expand2square(image, tuple(int(x*255) for x in vis_processor.image_mean))
        image_tensor = vis_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    trans_frames = []
    if not os.path.exists(image1) and os.path.exists(video1):
        video = cv2.VideoCapture(video1)
            
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
                pil_image = Image.fromarray(rgb_frame)
                rgb_frame = expand2square(pil_image, tuple(int(x*255) for x in vis_processor.image_mean))
                        
                video_frames.append(rgb_frame)
            cnt += 1
        video.release()
        for rgb in video_frames:
            trans_frames.append(vis_processor.preprocess(rgb, return_tensors='pt')['pixel_values'][0])
        trans_frames_ = torch.stack(trans_frames).to(torch.float16)

    # import pdb;pdb.set_trace()
    text_en_out, state_ = handler.generate([trans_frames_ if trans_frames else image_tensor], textbox_in, first_run=first_run, state=state_)
    state_.messages[-1] = (state_.roles[1], text_en_out)

    text_en_out = text_en_out.split('#')[0]
    textbox_out = text_en_out

    show_images = ""
    # import pdb;pdb.set_trace()
    if os.path.exists(image1):
        filename = save_image_to_local(image1)
        show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
    if os.path.exists(video1):
        filename = save_video_to_local(video1)
        show_images += f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={filename}"></video>'

    if flag:
        state.append_message(state.roles[0], textbox_in + "\n" + show_images)
    state.append_message(state.roles[1], textbox_out)

    return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), images_tensor, gr.update(value=image1 if os.path.exists(image1) else None, interactive=True), gr.update(value=video1 if os.path.exists(video1) else None, interactive=True))

def regenerate(state, state_):
    state.messages.pop(-1)
    state_.messages.pop(-1)
    if len(state.messages) > 0:
        return state, state_, state.to_gradio_chatbot(), False
    return (state, state_, state.to_gradio_chatbot(), True)


def clear_history(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True),
        gr.update(value=None, interactive=True),\
        gr.update(value=None, interactive=True),\
        True, state, state_, state.to_gradio_chatbot(), [[], []])



conv_mode = "omni_v1"
model_path = 'harvey2333/omni_video_assistant_5_3'
device = 'cuda'
load_8bit = False
load_4bit = False
dtype = torch.float16
handler = Chat(model_path, conv_mode=conv_mode, load_8bit=load_8bit, load_4bit=load_8bit, device=device)
# handler.model.to(dtype=dtype)
if not os.path.exists("temp"):
    os.makedirs("temp")

app = FastAPI()

textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
with gr.Blocks(title='🤖Omni: Video Assistant based on LLM', theme=gr.themes.Default(), css=block_css) as demo:
    gr.Markdown(title_markdown)
    state = gr.State()
    state_ = gr.State()
    first_run = gr.State()
    images_tensor = gr.State()
    with gr.Row():
        with gr.Column(scale=3):
            image1 = gr.Image(label="Input Image", type="filepath")
            video = gr.Video(label="Input Video")

            cur_dir = os.path.dirname(os.path.abspath(__file__))
            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/examples/extreme_ironing.jpg",
                        "What is unusual about this image?",
                    ],
                    [
                        f"{cur_dir}/examples/waterview.jpg",
                        "What are the things I should be cautious about when I visit here?",
                    ],
                    [
                        f"{cur_dir}/examples/desert.jpg",
                        "If there are factual errors in the questions, point it out; if not, proceed answering the question. What’s happening in the desert?",
                    ],
                ],
                inputs=[image1, textbox],
                fn=clear_history(gr.State(), gr.State())
            )
            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/examples/0A8CF.mp4",
                        "Please describe the following video in details.",
                    ],
                    [
                        f"{cur_dir}/examples/XJU8U.mp4",
                        "What does the character wear in the video?",
                    ],
                    [
                        f"{cur_dir}/examples/sample_demo_22.mp4",
                        "Describe the activity in the video.",
                    ],
                ],
                inputs=[video, textbox],
                fn=clear_history(gr.State(), gr.State())
            )

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="Omni-VideoAssistant", bubble_full_width=True).style(height=750)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(
                        value="Enter", variant="primary", interactive=True
                    )
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

    # with gr.Row():
        
    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)

    submit_btn.click(generate, [image1, video, textbox, first_run, state, state_, images_tensor],
                     [state, state_, chatbot, first_run, textbox, images_tensor, image1, video])

    regenerate_btn.click(regenerate, [state, state_], [state, state_, chatbot, first_run]).then(
        generate, [image1, video, textbox, first_run, state, state_, images_tensor], [state, state_, chatbot, first_run, textbox, images_tensor, image1, video])

    clear_btn.click(clear_history, [state, state_],
                    [image1, video, textbox, first_run, state, state_, chatbot, images_tensor])

# app = gr.mount_gradio_app(app, demo, path="/")
demo.launch()


# uvicorn llava.serve.gradio_web_server:app
