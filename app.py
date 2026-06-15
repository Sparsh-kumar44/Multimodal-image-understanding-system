import streamlit as st
from ultralytics import YOLO
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)
from PIL import Image

st.set_page_config(
    page_title="Image Understanding System",
    layout="centered"
)

@st.cache_resource
def load_models():

    detector = YOLO("yolov8n.pt")

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

    return (
        detector,
        caption_processor,
        caption_model,
        vqa_processor,
        vqa_model
    )


(
    detector,
    caption_processor,
    caption_model,
    vqa_processor,
    vqa_model
) = load_models()


def analyze(image, question):

    detection_results = detector(image)

    labels = set()

    for box in detection_results[0].boxes:
        labels.add(
            detector.names[int(box.cls)]
        )

    objects_found = (
        ", ".join(sorted(labels))
        if labels
        else "No objects detected"
    )

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

    answer = "No question provided"

    if question.strip():

        qa_inputs = vqa_processor(
            image,
            question,
            return_tensors="pt"
        )

        answer_ids = vqa_model.generate(
            **qa_inputs,
            max_new_tokens=10
        )

        answer = vqa_processor.decode(
            answer_ids[0],
            skip_special_tokens=True
        )

    return objects_found, caption, answer


st.title("Image Understanding System")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

question = st.text_input(
    "Ask a Question (Optional)"
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Analyze"):

        objects, caption, answer = analyze(
            image,
            question
        )

        st.subheader("Detected Objects")
        st.write(objects)

        st.subheader("Generated Caption")
        st.write(caption)

        st.subheader("Answer")
        st.write(answer)
