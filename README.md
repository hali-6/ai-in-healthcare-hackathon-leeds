# NTC — NAH to Cancer

An AI in Healthcare hackathon project that uses a deep-learning model to classify skin lesion images as **benign** or **malignant**, wrapped in a Streamlit web app with user accounts and scan history.

> ⚠️ **Research and educational prototype only.** NTC is not a medical device and is not clinically validated. Its predictions are not diagnoses and must not be used to make medical decisions. Do not use it with real patient data.

---

## Features

- **Analyse Image**: upload a JPG/PNG (or pick an example from the validation set) and get a benign/malignant prediction with confidence scores, inference time and image details. Includes zoom, rotate and remove controls.
- **Past Results**: every scan a user makes is saved to their account. This page shows the date and time, the image, and the benign/malignant scores for each one, newest first.
- **User accounts**: sign in with a username and password. Passwords are salted and hashed; scan images are encrypted before being stored.
- **About, Model Information and Usage Guide** pages.

## The model

| | |
|---|---|
| Architecture | ConvNeXt Tiny (`timm`), ImageNet-pretrained, fine-tuned for 2 classes |
| Input | RGB image resized to 224 × 224, ImageNet normalisation |
| Classes | `benign` = 0, `malignant` = 1 |
| Output | Softmax scores; a malignant score above 0.5 is classed as malignant |
| Training | 10 epochs, batch 32, AdamW (lr 1e-4, weight decay 0.05), cosine schedule, class-weighted cross-entropy |
| Augmentation | Random resized crop, horizontal/vertical flips, rotation ±30°, colour jitter |
| Checkpoint | The epoch with the best validation ROC-AUC is saved as `best_convnext_tiny.pth` |

## Project structure

```
skinmodel/
├── GUI/
│   ├── gui_main.py          # Streamlit app (all pages, styling, scan history)
│   └── login_page.py        # Sign-in screen and log-out
├── main.py                  # Trains the model and saves best_convnext_tiny.pth
├── run.py                   # Loads the model + predicts; shared by the app
├── database_aihack.py       # SQLite database: users, scans, predictions, audit log
├── user_data_storage.py     # Extra scan-record storage helpers
├── init_database.py         # Creates the database tables
├── create_dummy_user.py     # Creates a shared test account
├── data/
│   ├── train/{benign,malignant}/   # 6,289 benign / 5,590 malignant images
│   └── val/{benign,malignant}/     # 1,000 / 1,000 images
└── 6299.jpg                 # Sample image used by run.py
```

**Not in the repo** (see `.gitignore`): the model file `best_convnext_tiny.pth` (~111 MB), the database `melanoma_vault.db`, the encryption key `vault_master.key`, and virtual environments.

## Getting started

Run all commands from the `skinmodel` project folder.

### 1. Set up Python

Tested with Python 3.14.

```bash
python3 -m venv env1
source env1/bin/activate          # Windows: env1\Scripts\activate
pip install torch torchvision timm streamlit pillow numpy scikit-learn cryptography huggingface_hub
```

### 2. Add the model file

`best_convnext_tiny.pth` is too large for GitHub. Get a copy and put it in the project folder (next to `run.py`), or train your own (see [Training](#training)).

### 3. Create the database and a test account

```bash
python init_database.py
python create_dummy_user.py
```

This creates the test login **`testpatient01`** / **`TestPatient2026!`**.

### 4. Run the app

```bash
streamlit run GUI/gui_main.py
```

The app opens at http://localhost:8501. Sign in with the test account.

> Run Streamlit from the project folder, not from inside `GUI/`, so the app finds the database in the right place.

## Other scripts

**Quick model check** (predicts on `6299.jpg`):

```bash
python run.py
```

**Training**: the dataset must be in `data/train` and `data/val`, one folder per class. Training downloads the pretrained weights from Hugging Face, so set your token first:

```bash
export HF_TOKEN=your_token_here
python main.py
```

This overwrites `best_convnext_tiny.pth` with the new best model.

## Data storage and security

- All app data lives in a local SQLite file, `melanoma_vault.db`, created in the project folder.
- Passwords are stored as salted hashes and can't be recovered, only reset.
- Scan images and notes are encrypted with a key in `vault_master.key`, created automatically on first run. **Never commit or share this file publicly.** Without it, stored images can't be decrypted.
- **The database is not shared.** Each computer has its own database and accounts, and they don't sync through GitHub. For a demo where everyone sees the same data, run the app on one laptop and let others open the *Network URL* Streamlit prints, on the same Wi-Fi.

## Disclaimer

NTC — NAH to Cancer is a hackathon prototype built for learning. It is not a substitute for a doctor or dermatologist. If you're worried about a mole or a change in your skin, seek professional medical advice.

## Credits

Built by a team of four students at the AI in Healthcare Hackathon, Leeds.

| Name | GitHub |
|---|---|
| **Hashim Ali** | [@hali-6](https://github.com/hali-6) |
| **Natasha Chisvo** | [@Natasha-Chisvo](https://github.com/Natasha-Chisvo) |
| **Neil Suresh** | [@Neil-Suresh](https://github.com/Neil-Suresh) |
| **Ahyan Qadri** | [@MAhyanqadri](https://github.com/MAhyanqadri) |

*Four minds. One mission. NAH to Cancer.*
