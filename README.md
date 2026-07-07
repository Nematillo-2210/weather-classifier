# Weather Image Classifier

A deep learning model that classifies images into 11 weather conditions: dew, fogsmog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, and snow.

**Live App:** [Try it on Hugging Face Spaces](https://huggingface.co/spaces/nematillo-2210/weather-classifier)

---

## Dataset

- **Source:** Weather Image Recognition dataset from Kaggle
- **Classes:** 11 weather conditions
- **Size:** ~7,600 images total (232–1,160 images per class)

---

## Model Architecture

- **Base model:** ResNet34 pretrained on ImageNet
- **Framework:** Fast.ai
- **Approach:** Transfer learning — frozen backbone trained first, then full fine-tuning
- **Input size:** 64×64 pixels

---

## Training Results

| Strategy | Accuracy |
|---|---|
| fine_tune(5) | 89.8% |
| + fit_one_cycle(3, lr=1e-5) | no improvement |

**Final accuracy: 89.8%** on 11-class weather classification.

---

## Screenshot

*<img width="1630" height="682" alt="image" src="https://github.com/user-attachments/assets/aaf00e9b-d017-4aec-8503-f6f697b6d7ca" />
*

---

## How to Run Locally

```
pip install fastai gradio
python app.py
```

---
