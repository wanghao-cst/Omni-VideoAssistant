
# Omni-VideoAssistant
Training and Dataset will be released soon.
A more powerful model is on the way.

## 🔨 Preparation
```bash
git clone https://github.com/wanghao-cst/Omni-VideoAssistant
cd Omni-VideoAssistant
```
```shell
conda create -n omni python=3.10 -y
conda activate omni
pip install --upgrade pip
pip install -e .
```

## 🌟 Start here
### Download Omni Preview Model
Download for CLI inference only, gradio web UI will download it automatically.
[Omni Preview Model 5.3](https://huggingface.co/harvey2333/omni_video_assistant_5_3)

### Inference in Gradio Web UI

```Shell
CUDA_VISIBLE_DEVICES=0 python -m  llava.serve.gradio_demo
```
<p align="left">
<img src="assets/gradio_demo.png" width=100%>
</p>

### Inference in CLI
```
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.run_omni \
    --model-path "path to omni checkpoints" \
    --image-file "llava/serve/examples/extreme_ironing.jpg" \
    --query "What is unusual about this image?" \
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.run_omni \
    --model-path path to omni checkpoints \
    --video-file "llava/serve/examples/0A8CF.mp4" \
    --query "Describe the activity in the video" \
```

## Results Comparision
### Image understanding
<p align="left">
<img src="assets/val_img.png" width=100%>
</p>

### Video understanding
<p align="left">
<img src="assets/val_vid.png" width=100%>
</p>


## 😊 Acknowledgment

This work is based on [MVCE for unlimited training data generation.](https://github.com/shajiayu1/MVCE/), [LLaVA for pretrained model](https://github.com/haotian-liu/LLaVA/)
