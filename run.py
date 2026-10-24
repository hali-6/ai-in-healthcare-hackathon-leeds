import torch, timm
from PIL import Image
from torchvision import transforms as T

IMG_SIZE = 224
device = "mps" if torch.backends.mps.is_available() else "cpu"

# same preprocessing as validation (no augmentation)
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# rebuild the same architecture, then load the saved weights
model = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("best_convnext_tiny.pth", map_location=device))
model.to(device).eval()

img = Image.open("6299.jpg").convert("RGB")
x = tf(img).unsqueeze(0).to(device)   # add batch dimension -> [1, 3, 224, 224]

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]

classes = ["benign", "malignant"]     # matches class_to_idx: benign=0, malignant=1
print(f"benign: {probs[0]:.3f} | malignant: {probs[1]:.3f}")
print("Prediction:", classes[probs.argmax().item()])