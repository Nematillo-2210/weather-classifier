import gradio as gr
from fastai.vision.all import *
import pathlib
import platform

plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

learner = load_learner('weather_classifier.pkl')

def wrapper(img):
    img = PILImage.create(img)
    tuple = learner.predict(img)
    unpacked = dict(zip(learner.dls.vocab, map(float, tuple[2])))
    return unpacked

inface = gr.Interface(
    fn=wrapper,
    inputs=gr.Image(),
    outputs=gr.Label(),
    title='Weather Classifier',
    api_name=False
)

inface.launch()
