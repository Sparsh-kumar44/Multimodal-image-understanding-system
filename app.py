import streamlit as st
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)

st.set_page_config(page_title="Image Assistant")

@st.cache_resource
def load_models():

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
        caption_processor,
        caption_model,
        vqa_processor,
        vqa_model
    )


(
    caption_processor,
    caption_model,
    vqa_processor,
    vqa_model
) = load_models()


def generate_caption(image):

    inputs = caption_processor(
        image,
        return_tensors="pt"
    )

    output = caption_model.generate(
        **inputs,
        max_new_tokens=30
    )

    return caption_processor.decode(
        output[0],
        skip_special_tokens=True
    )


def answer_question(image, question):

    inputs = vqa_processor(
        image,
        question,
        return_tensors="pt"
    )

    output = vqa_model.generate(
        **inputs,
        max_new_tokens=10
    )

    return vqa_processor.decode(
        output[0],
        skip_special_tokens=True
    )


st.title("Visual AI Assistant")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

question = st.text_input(
    "Ask a question about the image"
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    if st.button("Analyze"):

        caption = generate_caption(image)

        st.subheader("Image Description")
        st.write(caption)

        if question.strip():

            answer = answer_question(
                image,
                question
            )

            st.subheader("Answer")
            st.write(answer)
