import os
import zipfile
import tempfile

import torch
import pandas as pd
import streamlit as st

from PIL import Image
from torchvision import transforms

from model import simplecnn, googlenetlike


class_names = [
    "mild_demented",
    "moderate_demented",
    "non_demented",
    "very_mild_demented"
]


device = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_model(model_name):
    if model_name == "simple cnn":
        model = simplecnn(num_classes=4)
        model_path = "simplecnn_model.pth"

    elif model_name == "googlenet-like cnn":
        model = googlenetlike(num_classes=4)
        model_path = "googlenetlike_model.pth"

    else:
        raise ValueError("invalid model name")

    if not os.path.exists(model_path):
        st.error(f"model file not found: {model_path}")
        st.stop()

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    return model


def predict_image(image, model):
    image = image.convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_index = predicted.item()
    predicted_class = class_names[predicted_index]
    confidence_value = confidence.item() * 100

    probs = probabilities[0].cpu().numpy()

    return predicted_class, confidence_value, probs


def is_image_file(filename):
    valid_extensions = [".jpg", ".jpeg", ".png"]
    filename = filename.lower()

    return any(filename.endswith(ext) for ext in valid_extensions)


st.set_page_config(
    page_title="alzheimer mri stage classification",
    layout="centered"
)


st.title("alzheimer mri stage classification")
st.write("computer-aided preliminary classification of brain mri images.")


model_name = st.sidebar.selectbox(
    "choose model",
    ["simple cnn", "googlenet-like cnn"]
)

model = load_model(model_name)


option = st.sidebar.radio(
    "choose prediction mode",
    ["single image prediction", "batch prediction"]
)


if option == "single image prediction":
    uploaded_file = st.file_uploader(
        "upload one mri image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="uploaded image",
            width=300
        )

        predicted_class, confidence_value, probs = predict_image(image, model)

        st.subheader("prediction result")
        st.write(f"model: {model_name}")
        st.write(f"predicted class: {predicted_class}")
        st.write(f"confidence: {confidence_value:.2f}%")

        st.subheader("class probabilities")

        prob_data = []

        for i, class_name in enumerate(class_names):
            prob_data.append({
                "class": class_name,
                "probability": probs[i] * 100
            })

        prob_df = pd.DataFrame(prob_data)
        st.dataframe(prob_df)

        st.warning(
            "this result is not a final medical diagnosis. "
            "it is a computer-aided preliminary classification."
        )

elif option == "batch prediction":
    upload_type = st.radio(
        "choose upload type",
        ["upload multiple images", "upload zip file"]
    )

    predictions = []

    if upload_type == "upload multiple images":
        uploaded_files = st.file_uploader(
            "upload mri images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)

                predicted_class, confidence_value, probs = predict_image(
                    image,
                    model
                )

                predictions.append({
                    "image name": uploaded_file.name,
                    "predicted class": predicted_class,
                    "confidence": confidence_value
                })

    elif upload_type == "upload zip file":
        uploaded_zip = st.file_uploader(
            "upload zip file containing mri images",
            type=["zip"]
        )

        if uploaded_zip is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "uploaded_images.zip")

                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())

                extract_dir = os.path.join(temp_dir, "images")

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

                image_paths = []

                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if is_image_file(file):
                            image_paths.append(os.path.join(root, file))

                if len(image_paths) == 0:
                    st.error("no image files found in the uploaded zip file.")

                for image_path in image_paths:
                    image = Image.open(image_path)

                    predicted_class, confidence_value, probs = predict_image(
                        image,
                        model
                    )

                    predictions.append({
                        "image name": os.path.basename(image_path),
                        "predicted class": predicted_class,
                        "confidence": confidence_value
                    })

    if len(predictions) > 0:
        results_df = pd.DataFrame(predictions)

        st.subheader("batch prediction results")
        st.write(f"model: {model_name}")
        st.dataframe(results_df)

        summary_df = (
            results_df["predicted class"]
            .value_counts()
            .reset_index()
        )

        summary_df.columns = ["predicted class", "count"]

        total_images = len(results_df)

        summary_df["percentage"] = (
            summary_df["count"] / total_images * 100
        )

        st.subheader("dataset distribution summary")
        st.dataframe(summary_df)

        dominant_class = summary_df.iloc[0]["predicted class"]
        dominant_percentage = summary_df.iloc[0]["percentage"]

        st.subheader("dominant predicted stage")
        st.write(f"dominant predicted stage: {dominant_class}")

        st.write(
            f"the uploaded dataset is mostly predicted as "
            f"{dominant_class}, representing "
            f"{dominant_percentage:.2f}% of the images."
        )

        csv_file = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="download prediction results as csv",
            data=csv_file,
            file_name="batch_predictions.csv",
            mime="text/csv"
        )

        st.warning(
            "these results are not final medical diagnoses. "
            "they are computer-aided preliminary classifications."
        )