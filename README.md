
# Omni-VideoAssistant
Author Shihao Wang and Dongyang Yu and Wangpeng An

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

### Inference in cmd
```
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.run_omni \
    --model-path "path to omni checkpoints" \
    --image-file "llava/serve/examples/extreme_ironing.jpg" \
    --query "What is unusual about this image?" \
    --image-aspect-ratio "pad"
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.run_omni \
    --model-path path to omni checkpoints \
    --video-file "llava/serve/examples/0A8CF.mp4" \
    --query "Describe the activity in the video" \
    --image-aspect-ratio "pad"
```



## 😊 Acknowledgment

This work is based on [MVCE](https://github.com/shajiayu1/MVCE/), [LLaVA](https://github.com/haotian-liu/LLaVA/)
