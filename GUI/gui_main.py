
import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms as T
from pathlib import Path

# Configuration
IMG_SIZE = 224
CLASSES = ["benign", "malignant"]
MODEL_PATH = Path("best_convnext_tiny.pth")

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Same preprocessing as run.py
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# Load model only once
@st.cache_resource
def load_model():
    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=2
    )

    weights = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    return model


# Make a prediction
def predict(image, model):
    image = image.convert("RGB")
    x = tf(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(x)
        probs = torch.softmax(output, dim=1)[0]

    return probs.cpu().tolist()


# GUI
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬"
)

st.title("Skin Lesion Classifier")
st.write("Upload an image to analyse it using ConvNeXt Tiny.")

st.warning(
    "Research prototype only. Predictions are not "
    "medical diagnoses and should not guide care."
)

uploaded_file = st.file_uploader(
    "Drag and drop an image here",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width="stretch"
        )

        if not MODEL_PATH.exists():
            st.error("Model weights not found.")
            st.stop()

        model = load_model()

        with st.spinner("Analysing image..."):
            probs = predict(image, model)

        predicted_class = CLASSES[
            max(range(len(probs)), key=lambda i: probs[i])
        ]

        st.subheader("Prediction Results")
        st.write("Predicted class:", predicted_class)

        st.write(f"Benign score: {probs[0]:.1%}")
        st.progress(probs[0])

        st.write(f"Malignant score: {probs[1]:.1%}")
        st.progress(probs[1])

        st.caption(
            "These are model softmax scores, not "
            "calibrated probabilities of disease."
        )

    except Exception as e:
        st.error(f"Unable to analyse image: {e}")
