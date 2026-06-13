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


try:
    with open(
        "evaluation_results.txt",
        "r",
        encoding="utf-8"
    ) as f:

        metrics_text = f.read()

except:

    metrics_text = "Run evaluate.py first."


with gr.Blocks() as demo:

    gr.Markdown(
        """
# Vision-Language Model for Image Understanding

YOLOv8 + BLIP + BLIP-VQA
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
            lines=12,
            label="YOLO Evaluation Results",
            interactive=False
        )

demo.launch()