import torch, timm
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

IMG_SIZE = 224
CLASSES = ["benign", "malignant"]  # matches class_to_idx: benign=0, malignant=1
MODEL_PATH = Path(__file__).resolve().parent / "best_convnext_tiny.pth"
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# same preprocessing as validation (no augmentation)
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(path=MODEL_PATH):
    """Rebuild the same architecture, then load the saved weights."""
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def predict(img, model):
    """Returns [benign, malignant] scores for a PIL image."""
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)  # add batch dimension -> [1, 3, 224, 224]
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return probs.cpu().tolist()


if __name__ == "__main__":
    # Only runs with `python run.py`, not when another file imports this one
    probs = predict(Image.open(MODEL_PATH.parent / "6299.jpg"), load_model())
    print(f"benign: {probs[0]:.3f} | malignant: {probs[1]:.3f}")
    print("Prediction:", CLASSES[probs.index(max(probs))])
