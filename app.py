import gradio as gr
from ultralytics import YOLO
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)

print("Loading models...")

yolo_model = YOLO("yolov8n.pt")

caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

vqa_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)

vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)

print("Models Loaded!")

metrics_text = """
Dataset : COCO128

Precision : 63.85 %

Recall : 53.61 %

mAP@50 : 60.54 %

mAP@50-95 : 44.54 %
"""


def analyze_image(image, question):

    if image is None:
        return "", "", ""

    results = yolo_model(image)

    detected_objects = []

    for box in results[0].boxes:
        cls_id = int(box.cls)
        detected_objects.append(
            yolo_model.names[cls_id]
        )

    detected_objects = list(set(detected_objects))

    if len(detected_objects) == 0:
        detected_text = "No objects detected"
    else:
        detected_text = ", ".join(detected_objects)

    caption_inputs = caption_processor(
        image,
        return_tensors="pt"
    )

    caption_ids = caption_model.generate(
        **caption_inputs,
        max_new_tokens=30
    )

    caption = caption_processor.decode(
        caption_ids[0],
        skip_special_tokens=True
    )

    answer = ""

    if question and question.strip():

        vqa_inputs = vqa_processor(
            image,
            question,
            return_tensors="pt"
        )

        answer_ids = vqa_model.generate(
            **vqa_inputs,
            max_new_tokens=10
        )

        answer = vqa_processor.decode(
            answer_ids[0],
            skip_special_tokens=True
        )

    return detected_text, caption, answer


with gr.Blocks(
    title="Vision-Language Model for Image Understanding"
) as demo:

    gr.Markdown(
        """
# Vision-Language Model for Image Understanding

### YOLOv8 + BLIP + BLIP-VQA

This project combines object detection,
image captioning, and visual question answering
within a single multimodal AI application.
"""
    )

    with gr.Tab("Image Analysis"):

        image_input = gr.Image(
            type="pil",
            label="Upload Image"
        )

        question_input = gr.Textbox(
            label="Ask a Question",
            placeholder="What animal is shown in the image?"
        )

        analyze_btn = gr.Button(
            "Analyze Image"
        )

        clear_btn = gr.ClearButton(
            [
                image_input,
                question_input
            ]
        )

        objects_output = gr.Textbox(
            label="Detected Objects"
        )

        caption_output = gr.Textbox(
            label="Generated Caption"
        )

        answer_output = gr.Textbox(
            label="Answer"
        )

        analyze_btn.click(
            fn=analyze_image,
            inputs=[
                image_input,
                question_input
            ],
            outputs=[
                objects_output,
                caption_output,
                answer_output
            ]
        )

    with gr.Tab("Performance Evaluation"):

        gr.Textbox(
            value=metrics_text,
            lines=10,
            label="YOLO Evaluation Results",
            interactive=False
        )

demo.launch()