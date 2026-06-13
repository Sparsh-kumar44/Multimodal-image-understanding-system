# Vision-Language Model for Image Understanding

A multimodal AI system that combines:

* YOLOv8 Object Detection
* BLIP Image Captioning
* BLIP Visual Question Answering (VQA)

## Features

* Detect objects in uploaded images
* Generate natural language captions
* Answer questions related to image content
* Interactive Gradio web interface

## Technologies Used

* Python
* PyTorch
* Transformers
* YOLOv8
* BLIP
* BLIP-VQA
* Gradio

## Project Pipeline

Input Image
↓
YOLOv8 → Detected Objects

Input Image
↓
BLIP → Generated Caption

Input Image + Question
↓
BLIP-VQA → Answer

## Installation

pip install -r requirements.txt

## Run Evaluation

python evaluate.py

## Launch Application

python app.py

## Example Output

Detected Objects:
dog

Generated Caption:
A brown and white dog is running on the beach.

Question:
What animal is shown in the image?

Answer:
dog

## Author

Sparsh Kumar
