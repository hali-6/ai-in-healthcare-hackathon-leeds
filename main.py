import torch, timm, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
from huggingface_hub import login

import os
token = os.environ.get("HF_TOKEN")

login(token)


# ---- config ----
DATA_DIR = "data"
IMG_SIZE = 224
BATCH = 32
EPOCHS = 10
LR = 1e-4

def main():
    device = ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")
    print("Using", device)

    # ---- data ----
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.ToTensor(), norm,
    ])
    val_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), norm])

    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", train_tf)
    val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", val_tf)
    print("Classes:", train_ds.class_to_idx)

    train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, BATCH, num_workers=2)

    # ---- class weights for imbalance ----
    counts = np.bincount(train_ds.targets)
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ---- model ----
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    # ---- train / eval ----
    def evaluate():
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y in val_dl:
                p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
                probs += p.cpu().tolist()
                labels += y.tolist()
        probs, labels = np.array(probs), np.array(labels)
        preds = (probs > 0.5).astype(int)
        auc = roc_auc_score(labels, probs)
        sens = recall_score(labels, preds)  # recall on class 1
        return auc, sens, confusion_matrix(labels, preds)

    best_auc = 0
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for i, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            print(i)
        scheduler.step()

        auc, sens, cm = evaluate()
        print(f"Epoch {epoch+1}/{EPOCHS} | loss {total/len(train_ds):.4f} | "
            f"val AUC {auc:.3f} | sensitivity {sens:.3f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_convnext_tiny.pth")

    print("Best val AUC:", best_auc)
    print("Confusion matrix (last epoch):\n", cm)


if __name__ == "__main__":
    main()
    