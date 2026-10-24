"""NTC — NAH to Cancer · Streamlit app.

Run from the project root:  streamlit run GUI/gui_main.py
"""
import base64
import hashlib
import io
import random
import sys
import time
from html import escape
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

# =============================================================================
# Configuration
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent  # project root (one level above GUI/)
sys.path.insert(0, str(ROOT))  # so we can import run.py from the project root

import run  # the model, preprocessing and prediction all come from run.py
from run import CLASSES, IMG_SIZE, MODEL_PATH, device

# These need the project root on the path too (database_aihack.py lives there)
import database_aihack as db
from login_page import logout, render_login

DATA_DIR = ROOT / "data"
MIN_LOADING_SECONDS = 1.2  # keep the loading state visible long enough to notice


# =============================================================================
# Model (from run.py)
# =============================================================================
@st.cache_resource
def load_model():
    return run.load_model()  # cached so the model loads once, not on every click


def predict(image, model):
    """Returns ([benign, malignant] scores, inference time in seconds)."""
    start = time.perf_counter()
    probs = run.predict(image, model)
    return probs, time.perf_counter() - start


# =============================================================================
# Example images (from data/val)
# =============================================================================
@st.cache_data
def list_images(cls):
    return sorted(str(p) for p in (DATA_DIR / "val" / cls).glob("*.jpg"))


def pick_examples(seed):
    benign, malignant = list_images("benign"), list_images("malignant")
    if len(benign) < 2 or len(malignant) < 2:
        return []
    rng = random.Random(seed)
    b, m = rng.sample(benign, 2), rng.sample(malignant, 2)
    return [(b[0], "Benign"), (m[0], "Malignant"), (b[1], "Benign"), (m[1], "Malignant")]


@st.cache_data
def square_thumb(path, size=260):
    return ImageOps.fit(Image.open(path).convert("RGB"), (size, size))


# =============================================================================
# Icons (Lucide-style line icons, inlined as SVG)
# =============================================================================
ICONS = {
    "flask": '<path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "eye": '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "chip": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>',
    "x": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/>',
}


def ribbon_svg(height=64):
    """Filled cancer-awareness ribbon (one looped band, right tail drawn over the left)."""
    return f"""
<svg width="{height * 0.62:.0f}" height="{height}" viewBox="0 0 62 100" fill="none" aria-hidden="true">
  <defs>
    <linearGradient id="rb" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#fbcfe8"/><stop offset="0.45" stop-color="#f472b6"/><stop offset="1" stop-color="#be185d"/>
    </linearGradient>
  </defs>
  <path d="M31 56 C40 42 46 32 44 22 C42 10 20 10 18 22 C16 32 22 42 31 56 L14 92"
        stroke="url(#rb)" stroke-width="10" stroke-linejoin="round"/>
  <path d="M31 56 L48 92" stroke="#00000055" stroke-width="12"/>
  <path d="M31 56 L48 92" stroke="url(#rb)" stroke-width="10"/>
</svg>"""


def icon(name, size=20, stroke=1.8, fill="none"):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="{fill}" '
        f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


# =============================================================================
# Styles
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Caveat:wght@400;500&display=swap');

:root {
  --bg: #120a1c;
  --card: rgba(30, 17, 45, 0.72);
  --card-2: rgba(42, 24, 62, 0.65);
  --line: rgba(168, 85, 247, 0.20);
  --line-2: rgba(192, 132, 252, 0.38);
  --pink: #e15cf2;
  --pink-2: #f0abfc;
  --violet: #a855f7;
  --lav: #b8a4f5;
  --green: #4ade80;
  --rose: #fb7185;
  --text: #f5eefe;
  --muted: #bfaed3;
  --font: 'Outfit', 'Avenir Next', 'Segoe UI', system-ui, sans-serif;
  --script: 'Caveat', 'Bradley Hand', 'Segoe Script', cursive;
}

/* ---------- page & sidebar ---------- */
html, body, .stApp, .stApp p, .stApp span, .stApp div, .stApp button, .stApp li { font-family: var(--font); }
/* keep Streamlit's icon font, or icons show up as their names ("info", "keyboard_double_arrow_left"…) */
.stApp [data-testid="stIconMaterial"], .stApp [class*="material-symbols"] { font-family: 'Material Symbols Rounded' !important; }
.stApp {
  background:
    radial-gradient(1100px 600px at 70% -10%, rgba(126, 34, 206, 0.22), transparent 60%),
    radial-gradient(800px 500px at 100% 100%, rgba(192, 38, 211, 0.10), transparent 60%),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1400px; padding: 1.6rem 2rem 2.5rem; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #2a1242 0%, #1f0e31 55%, #190b28 100%);
  border-right: 1px solid var(--line);
  min-width: 270px !important;
  max-width: 270px !important;
}
[data-testid="stSidebarUserContent"] { padding: 1.2rem 0.8rem 1.5rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem; }

/* sidebar navigation buttons */
[data-testid="stSidebar"] .stButton button {
  justify-content: flex-start;
  gap: 0.9rem;
  padding: 0.8rem 1rem;
  border-radius: 10px;
  border: 1px solid transparent;
  background: transparent;
  color: #ece3f7;
  font-size: 1.05rem;
  font-weight: 400;
  min-height: 3.1rem;
  transition: background 0.2s, border-color 0.2s;
}
[data-testid="stSidebar"] .stButton button p { font-size: 1.05rem; }
[data-testid="stSidebar"] .stButton button:hover { background: rgba(168, 85, 247, 0.12); color: #fff; }
[data-testid="stSidebar"] .stButton button[kind="primary"] {
  background: linear-gradient(90deg, rgba(147, 51, 234, 0.75), rgba(126, 34, 206, 0.55));
  border-color: rgba(192, 132, 252, 0.45);
  font-weight: 600;
  box-shadow: 0 6px 22px rgba(126, 34, 206, 0.35);
}
[data-testid="stSidebar"] .stButton button[kind="primary"] p { font-weight: 600; }
[data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"] { font-size: 1.45rem; color: #e9d5ff; }

/* ---------- main-area buttons ---------- */
[data-testid="stMain"] .stButton button[kind="tertiary"] { color: var(--text); }
[data-testid="stMain"] .stButton button[kind="tertiary"]:hover { color: var(--pink-2); }
.st-key-view_more .stButton button, .st-key-view_more button { color: var(--pink) !important; font-size: 0.9rem; }
.st-key-view_more { display: flex; justify-content: flex-end; }
.st-key-img_actions [data-testid="stHorizontalBlock"] { border-top: 1px solid var(--line); padding-top: 0.6rem; }
.st-key-img_actions [data-testid="stColumn"] { display: flex; justify-content: center; }
.st-key-img_actions [data-testid="stColumn"] + [data-testid="stColumn"] { border-left: 1px solid var(--line); }
.st-key-img_actions button [data-testid="stIconMaterial"] { font-size: 1.35rem; }

/* ---------- cards (keyed Streamlit containers) ---------- */
.st-key-card_drop, .st-key-card_examples, .st-key-card_image, .st-key-card_results {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1.1rem 1.2rem 1.2rem;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
}
.st-key-card_examples img { border-radius: 8px; }
.st-key-card_examples [data-testid="stColumn"] .stButton { display: flex; justify-content: center; }
.st-key-card_examples [data-testid="stColumn"] .stButton button p { font-weight: 500; }
.st-key-card_image img { border-radius: 10px; aspect-ratio: 5 / 4; object-fit: cover; width: 100%; }

/* ---------- file uploader, restyled to the mockup ---------- */
[data-testid="stFileUploaderDropzone"] {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  min-height: 15rem;
  padding: 1.6rem 1.2rem;
  background: rgba(20, 11, 32, 0.55);
  border: 1.5px dashed rgba(192, 132, 252, 0.42);
  border-radius: 12px;
  text-align: center;
  color: var(--text);
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--pink); }
[data-testid="stFileUploaderDropzone"]::before {
  content: "Drag and drop an image here";
  order: 0;
  padding-top: 3.4rem;
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text);
  background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='46' height='46' viewBox='0 0 24 24' fill='none' stroke='%23e15cf2' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 14.9A7 7 0 1 1 15.7 8h1.8a4.5 4.5 0 0 1 2.5 8.2'/%3E%3Cpath d='M12 12v9'/%3E%3Cpath d='m16 16-4-4-4 4'/%3E%3C/svg%3E") top center / 46px no-repeat;
}
[data-testid="stFileUploaderDropzone"]::after {
  content: "or click to browse";
  order: 1;
  margin-top: -0.4rem;
  font-size: 0.95rem;
  color: var(--muted);
}
[data-testid="stFileUploaderDropzone"] > div { order: 2; }
[data-testid="stFileUploaderDropzone"] > [data-testid="stFileUploaderDropzoneInstructions"] { order: 3; }
[data-testid="stFileUploaderDropzone"] button {
  background: linear-gradient(90deg, #c026d3, #d946ef);
  border: none;
  color: #1b0b26;
  border-radius: 10px;
  padding: 0.55rem 3.6rem;
  font-weight: 600;
  box-shadow: 0 6px 20px rgba(217, 70, 239, 0.35);
}
[data-testid="stFileUploaderDropzone"] button:hover { filter: brightness(1.08); color: #1b0b26; }
[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] { display: none; }
[data-testid="stFileUploaderDropzone"] button p { font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] button p::after { content: "Browse Files"; font-size: 1rem; font-weight: 600; }
[data-testid="stFileUploaderDropzoneInstructions"] {
  width: 85%;
  border-top: 1px solid var(--line);
  padding-top: 0.8rem;
  margin-top: 0.3rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] * { font-size: 0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"]::after {
  content: "Supported formats: JPG, JPEG, PNG\\A Recommended: clear, high-resolution images";
  white-space: pre-line;
  font-size: 0.82rem;
  line-height: 1.5;
  color: var(--muted);
}
[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span { color: var(--text); }

/* ---------- typography & shared pieces ---------- */
.ntc-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 2rem; flex-wrap: wrap; margin-bottom: 0.6rem; }
.ntc-head-main { flex: 1 1 520px; min-width: 0; }
.ntc-kicker-row { display: flex; align-items: center; gap: 1.2rem; }
.ntc-kicker { font-size: 0.85rem; letter-spacing: 0.32em; text-transform: uppercase; color: var(--pink-2); white-space: nowrap; }
.ntc-kicker-line { flex: 1; height: 1px; background: linear-gradient(90deg, var(--line-2), transparent); }
.ntc-pill {
  display: inline-flex; align-items: center; gap: 0.55rem;
  border: 1px solid var(--line-2); border-radius: 999px;
  padding: 0.45rem 1.1rem; font-size: 0.72rem; letter-spacing: 0.12em;
  color: var(--pink-2); text-transform: uppercase; white-space: nowrap;
  background: rgba(126, 34, 206, 0.12);
}
.ntc-title { font-size: clamp(2rem, 3.6vw, 3.1rem); font-weight: 700; color: #fff; margin: 0.9rem 0 0.4rem; line-height: 1.1; letter-spacing: -0.01em; }
.ntc-title em { font-style: normal; color: var(--pink); }
.ntc-sub { font-size: 1.25rem; color: #e5dbf0; margin: 0; font-weight: 300; max-width: 46rem; }
.ntc-head-side { display: flex; flex-direction: column; align-items: flex-end; gap: 1.2rem; padding-top: 0.1rem; }
.ntc-script {
  font-family: var(--script); font-size: 2.3rem; color: var(--pink-2);
  transform: rotate(-9deg); text-align: right; line-height: 0.95;
  padding: 0.4rem 0.6rem 0.8rem 0; position: relative;
}
.ntc-script::after { content: ""; position: absolute; right: 0.4rem; bottom: 0.15rem; width: 75%; height: 2px; border-radius: 2px; background: linear-gradient(90deg, transparent, var(--pink)); }
.ntc-script span { display: block; padding-right: 1rem; }

.ntc-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1.3rem 1.4rem;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
  color: var(--text);
}
.ntc-card-head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin-bottom: 0.9rem; }
.ntc-card-title { font-size: 1.1rem; font-weight: 600; color: #fff; margin: 0; }
.ntc-muted { color: var(--muted); font-size: 0.95rem; }
.ntc-h2 { font-size: 1.75rem; font-weight: 600; color: #fff; margin: 0 0 0.9rem; position: relative; padding-bottom: 0.7rem; }
.ntc-h2::after { content: ""; position: absolute; left: 0; bottom: 0; width: 2.4rem; height: 2px; background: var(--pink); border-radius: 2px; }
.ntc-body p { color: #e7dcf3; font-size: 1rem; line-height: 1.55; margin: 0 0 0.75rem; font-weight: 300; }

/* ---------- analyse page ---------- */
.ntc-empty {
  aspect-ratio: 5 / 4; border-radius: 10px; border: 1px dashed var(--line-2);
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.6rem;
  color: var(--muted); background: rgba(20, 11, 32, 0.45); text-align: center; padding: 1rem;
}
.ntc-empty svg { color: var(--violet); }
.ntc-status { display: inline-flex; align-items: center; gap: 0.5rem; font-size: 0.9rem; color: var(--green); }
.ntc-status::before { content: ""; width: 0.6rem; height: 0.6rem; border-radius: 50%; background: var(--green); box-shadow: 0 0 10px var(--green); }
.ntc-status.wait { color: var(--pink-2); }
.ntc-status.wait::before { background: var(--pink); box-shadow: 0 0 10px var(--pink); animation: ntc-blink 1s ease-in-out infinite; }
.ntc-status.idle { color: var(--muted); }
.ntc-status.idle::before { background: var(--muted); box-shadow: none; }
@keyframes ntc-blink { 50% { opacity: 0.25; } }

.ntc-pred {
  display: flex; align-items: center; gap: 1.1rem;
  border: 1px solid var(--line-2); border-radius: 10px; padding: 1.1rem 1.3rem;
  background: linear-gradient(90deg, rgba(126, 34, 206, 0.22), rgba(42, 24, 62, 0.4));
}
.ntc-pred.malignant { border-color: rgba(251, 113, 133, 0.55); background: linear-gradient(90deg, rgba(190, 18, 60, 0.22), rgba(42, 24, 62, 0.4)); }
.ntc-pred > svg { color: var(--pink); flex: none; }
.ntc-pred.malignant > svg { color: var(--rose); }
.ntc-pred-main { flex: 1; }
.ntc-pred small { display: block; font-size: 0.8rem; color: var(--muted); }
.ntc-pred strong { font-size: 2rem; font-weight: 600; color: var(--pink-2); line-height: 1.15; }
.ntc-pred.malignant strong { color: #fda4af; }
.ntc-pred-conf { text-align: center; border-left: 1px solid var(--line-2); padding-left: 1.3rem; }
.ntc-pred-conf strong { color: #fff; font-size: 1.6rem; }

.ntc-bar { margin-top: 1.1rem; }
.ntc-bar label { display: block; font-size: 0.95rem; color: #fff; margin-bottom: 0.35rem; }
.ntc-bar-row { display: flex; align-items: center; gap: 1rem; }
.ntc-track { flex: 1; height: 13px; border-radius: 4px; background: rgba(255, 255, 255, 0.08); overflow: hidden; }
.ntc-track i { display: block; height: 100%; border-radius: 4px; animation: ntc-grow 0.9s ease-out both; }
.ntc-track i.benign { background: linear-gradient(90deg, #c026d3, #f0abfc); }
.ntc-track i.malignant { background: linear-gradient(90deg, #8b7ae6, #c4b5fd); }
.ntc-bar-row b { width: 3.6rem; text-align: right; font-weight: 500; color: #fff; }
.ntc-bar-row b.top { color: var(--pink); }
@keyframes ntc-grow { from { width: 0; } }

.ntc-details { margin-top: 1.3rem; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.ntc-details header { display: flex; align-items: center; gap: 0.6rem; padding: 0.75rem 0.9rem; font-weight: 500; color: #fff; border-bottom: 1px solid var(--line); }
.ntc-details header svg { color: var(--muted); }
.ntc-details table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.ntc-details td { padding: 0.55rem 0.9rem; border-top: 1px solid rgba(168, 85, 247, 0.12); color: #e9e0f5; }
.ntc-details tr:first-child td { border-top: 0; }
.ntc-details td:first-child { color: #fff; font-weight: 500; width: 46%; }
.ntc-details tr:nth-child(odd) td { background: rgba(255, 255, 255, 0.02); }

.ntc-loading { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1rem; min-height: 22rem; text-align: center; }
.ntc-spinner { width: 64px; height: 64px; border-radius: 50%; border: 3px solid rgba(168, 85, 247, 0.18); border-top-color: var(--pink); border-right-color: var(--violet); animation: ntc-spin 0.9s linear infinite; box-shadow: 0 0 30px rgba(225, 92, 242, 0.25); }
@keyframes ntc-spin { to { transform: rotate(360deg); } }
.ntc-loading strong { font-size: 1.15rem; color: #fff; font-weight: 500; }
.ntc-shimmer { width: 80%; height: 10px; border-radius: 6px; background: linear-gradient(90deg, rgba(168,85,247,0.1), rgba(240,171,252,0.5), rgba(168,85,247,0.1)); background-size: 200% 100%; animation: ntc-shim 1.2s linear infinite; }
@keyframes ntc-shim { from { background-position: 200% 0; } to { background-position: -200% 0; } }

.ntc-tips h3 { display: flex; align-items: center; gap: 0.7rem; font-size: 1.1rem; font-weight: 600; color: #fff; margin: 0 0 1.1rem; }
.ntc-tips h3 svg { color: var(--pink); }
.ntc-tips ul { list-style: none; padding: 0; margin: 0; }
.ntc-tips li { display: flex; gap: 0.8rem; align-items: flex-start; color: #e7dcf3; font-size: 0.95rem; margin: 0 0 1.05rem; }
.ntc-tips li svg { color: var(--muted); flex: none; margin-top: 0.1rem; }
.ntc-warn { display: flex; gap: 0.9rem; margin-top: 1rem; }
.ntc-warn > svg { color: var(--pink); flex: none; }
.ntc-warn strong { display: block; color: var(--pink-2); font-weight: 500; font-size: 0.95rem; margin-bottom: 0.6rem; line-height: 1.4; }
.ntc-warn p { color: var(--muted); font-size: 0.82rem; margin: 0; line-height: 1.5; }

/* ---------- about / info pages ---------- */
.ntc-grid { display: grid; gap: 1.1rem; margin-top: 0.8rem; }
.ntc-grid.two { grid-template-columns: 1fr 1fr; }
.ntc-cause { display: grid; grid-template-columns: 1.15fr 1px 1fr; gap: 2rem; }
.ntc-vrule { background: var(--line); }
.ntc-feature { display: flex; gap: 1.4rem; align-items: flex-start; margin-bottom: 1.2rem; }
.ntc-feature:last-child { margin-bottom: 0; }
.ntc-orb { flex: none; width: 70px; height: 70px; border-radius: 50%; display: grid; place-items: center; color: var(--pink); background: radial-gradient(circle at 35% 30%, rgba(168, 85, 247, 0.35), rgba(60, 25, 90, 0.75)); }
.ntc-orb.sm { width: 52px; height: 52px; }
.ntc-feature h4 { margin: 0.15rem 0 0.3rem; font-size: 1.1rem; font-weight: 600; color: #fff; }
.ntc-feature p { margin: 0; color: #e2d6ee; font-size: 0.97rem; line-height: 1.45; font-weight: 300; }
.ntc-motto { display: inline-flex; align-items: center; gap: 0.9rem; border: 1px solid var(--line-2); background: rgba(126, 34, 206, 0.16); border-radius: 999px; padding: 0.7rem 1.4rem 0.7rem 0.8rem; margin-top: 0.4rem; color: #fff; }
.ntc-motto i { display: grid; place-items: center; width: 2rem; height: 2rem; border-radius: 50%; background: var(--pink); color: #2a0f38; }
.ntc-motto b { color: var(--pink); font-weight: 500; }
.ntc-values .ntc-feature { gap: 1.1rem; margin-bottom: 0.9rem; align-items: center; }
.ntc-values .ntc-feature h4 { font-size: 1.02rem; margin: 0 0 0.1rem; }
.ntc-values .ntc-feature p { font-size: 0.93rem; }
.ntc-foot { display: flex; align-items: center; gap: 0.9rem; margin-top: 1.1rem; padding: 0.9rem 1.2rem; font-size: 0.9rem; color: #e2d6ee; }
.ntc-foot svg { color: var(--pink); flex: none; }
.ntc-kv { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
.ntc-kv td { padding: 0.6rem 0.2rem; border-top: 1px solid rgba(168, 85, 247, 0.14); color: #e9e0f5; vertical-align: top; }
.ntc-kv tr:first-child td { border-top: 0; }
.ntc-kv td:first-child { color: var(--muted); width: 42%; }
.ntc-steps { counter-reset: s; list-style: none; padding: 0; margin: 0; }
.ntc-steps li { counter-increment: s; display: flex; gap: 1rem; margin-bottom: 1.1rem; color: #e7dcf3; line-height: 1.5; }
.ntc-steps li::before { content: counter(s); flex: none; width: 2rem; height: 2rem; border-radius: 50%; display: grid; place-items: center; font-weight: 600; color: #2a0f38; background: linear-gradient(135deg, var(--pink-2), var(--pink)); }
.ntc-steps b { color: #fff; font-weight: 600; }

/* ---------- sidebar pieces ---------- */
.ntc-logo { padding: 0.4rem 0.9rem 1.3rem; border-bottom: 1px solid var(--line); margin: 0 -0.8rem 0.9rem; padding-left: 1.7rem; }
.ntc-logo .big { font-size: 3.4rem; font-weight: 800; line-height: 1; color: #fff; letter-spacing: -0.02em; }
.ntc-logo .big span { background: linear-gradient(135deg, var(--pink-2), var(--violet)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.ntc-logo .name { font-size: 1.6rem; font-weight: 700; color: #fff; margin-top: 0.3rem; }
.ntc-logo .name span { color: var(--pink); }
.ntc-logo hr { width: 1.4rem; border: 0; height: 2px; background: var(--pink); margin: 0.5rem 0 0.7rem; }
.ntc-logo p { margin: 0; color: #d8cbe6; font-size: 0.98rem; line-height: 1.35; }
.ntc-side-foot { margin: 2.5rem -0.8rem 0; }
.ntc-side-foot svg.wave { display: block; width: 100%; height: auto; }
.ntc-side-foot .ribbon { display: flex; gap: 0.9rem; align-items: center; padding: 1.4rem 1.6rem 0; color: var(--pink-2); font-size: 1.02rem; line-height: 1.35; }
.ntc-side-foot .ribbon svg { color: var(--pink); flex: none; }
.ntc-side-foot .ribbon b { color: var(--pink-2); font-weight: 400; }
.ntc-side-foot .ribbon span { color: #efe6f8; }
.ntc-side-foot hr { width: 2.2rem; border: 0; height: 2px; background: var(--pink); margin: 1.4rem 1.6rem 0.8rem; }
.ntc-side-foot p { margin: 0 1.6rem; color: #d8cbe6; font-size: 0.92rem; line-height: 1.45; }

/* ---------- user bar ---------- */
.st-key-userbar {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.45rem 0.9rem;
  margin-bottom: 1.1rem;
}
.ntc-user { display: flex; align-items: center; gap: 0.75rem; color: var(--muted); font-size: 1rem; }
.ntc-user b { color: #fff; font-weight: 700; }
.ntc-avatar {
  width: 2.1rem; height: 2.1rem; border-radius: 50%; display: grid; place-items: center;
  font-weight: 700; color: #2a0f38; background: linear-gradient(135deg, var(--pink-2), var(--pink));
}
.st-key-userbar .stButton button { color: var(--pink-2) !important; justify-content: flex-end; }

/* ---------- past results ---------- */
.ntc-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 0.6rem 0 1.1rem; }
.ntc-stats .ntc-card { padding: 1rem 1.2rem; }
.ntc-stats small { display: block; color: var(--muted); font-size: 0.85rem; margin-bottom: 0.25rem; }
.ntc-stats strong { font-size: 1.6rem; font-weight: 600; color: #fff; }
.ntc-hist { display: flex; flex-direction: column; gap: 0.9rem; }
.ntc-hist-row { display: grid; grid-template-columns: 110px 190px 1fr; gap: 1.4rem; align-items: center; padding: 1rem 1.2rem; }
.ntc-hist-img img, .ntc-hist-noimg { width: 110px; height: 110px; border-radius: 10px; object-fit: cover; display: block; }
.ntc-hist-noimg { display: grid; place-items: center; color: var(--muted); border: 1px dashed var(--line-2); }
.ntc-hist-date { font-size: 1.15rem; font-weight: 600; color: #fff; }
.ntc-hist-time { color: var(--pink-2); font-size: 0.95rem; margin-bottom: 0.3rem; }
.ntc-hist-name { font-size: 0.82rem; margin-bottom: 0.55rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ntc-badge { display: inline-block; padding: 0.2rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
.ntc-badge.benign { color: var(--pink-2); background: rgba(225, 92, 242, 0.15); border: 1px solid rgba(225, 92, 242, 0.45); }
.ntc-badge.malignant { color: #fda4af; background: rgba(251, 113, 133, 0.14); border: 1px solid rgba(251, 113, 133, 0.5); }
.ntc-hist-bar { display: grid; grid-template-columns: 5.5rem 1fr 3.8rem; gap: 0.9rem; align-items: center; margin: 0.45rem 0; }
.ntc-hist-bar span { color: #fff; font-size: 0.95rem; }
.ntc-hist-bar b { text-align: right; font-weight: 500; color: #fff; }

@media (max-width: 900px) {
  .ntc-stats { grid-template-columns: 1fr 1fr; }
  .ntc-hist-row { grid-template-columns: 90px 1fr; }
  .ntc-hist-img img, .ntc-hist-noimg { width: 90px; height: 90px; }
  .ntc-hist-bars { grid-column: 1 / -1; }
  .ntc-grid.two, .ntc-cause { grid-template-columns: 1fr; }
  .ntc-vrule { display: none; }
  .ntc-head-side { align-items: flex-start; }
}
</style>
"""

WAVE_SVG = """
<svg class="wave" viewBox="0 0 270 230" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <linearGradient id="wg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#f0abfc"/><stop offset="0.5" stop-color="#d946ef"/><stop offset="1" stop-color="#6b21a8"/>
    </linearGradient>
    <linearGradient id="wf" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#e879f9" stop-opacity="0.55"/><stop offset="1" stop-color="#581c87" stop-opacity="0.05"/>
    </linearGradient>
  </defs>
  <path d="M0 60 C60 10 110 150 170 120 S250 40 270 70 L270 230 L0 230 Z" fill="url(#wf)" opacity="0.55"/>
  <path d="M0 70 C60 20 110 160 170 128 S250 50 270 80" stroke="url(#wg)" stroke-width="1.4" opacity="0.9"/>
  <path d="M0 82 C60 32 112 170 172 138 S250 62 270 92" stroke="url(#wg)" stroke-width="1" opacity="0.7"/>
  <path d="M0 94 C62 44 114 178 174 148 S250 74 270 104" stroke="url(#wg)" stroke-width="0.8" opacity="0.55"/>
  <path d="M0 106 C64 56 116 186 176 158 S250 86 270 116" stroke="url(#wg)" stroke-width="0.7" opacity="0.4"/>
  <path d="M0 118 C66 68 118 194 178 168 S250 98 270 128" stroke="url(#wg)" stroke-width="0.6" opacity="0.3"/>
  <path d="M0 130 C68 80 120 200 180 178 S250 110 270 140" stroke="url(#wg)" stroke-width="0.5" opacity="0.22"/>
</svg>
"""


# =============================================================================
# HTML builders
# =============================================================================
def page_header(kicker, title, subtitle):
    return f"""
<div class="ntc-head">
  <div class="ntc-head-main">
    <div class="ntc-kicker-row"><span class="ntc-kicker">{kicker}</span><span class="ntc-kicker-line"></span></div>
    <h1 class="ntc-title">{title}</h1>
    <p class="ntc-sub">{subtitle}</p>
  </div>
  <div class="ntc-head-side">
    <span class="ntc-pill">{icon("flask", 15)} Research prototype</span>
    <div class="ntc-script">Early detection<span>saves lives.</span></div>
  </div>
</div>
"""


def sidebar_logo():
    return """
<div class="ntc-logo">
  <div class="big">NT<span>C</span></div>
  <div class="name"><span>NAH</span> to Cancer</div>
  <hr>
  <p>Early Detection.<br>Brighter Tomorrows.</p>
</div>
"""


def sidebar_footer():
    return f"""
<div class="ntc-side-foot">
  {WAVE_SVG}
  <div class="ribbon">{ribbon_svg(64)}<div><span>A future with</span><br><b>fewer skin cancer<br>cases is possible.</b></div></div>
  <hr>
  <p>Technology with purpose.<br>Awareness with impact.</p>
</div>
"""


def empty_image_html():
    return f"""
<div class="ntc-empty">{icon("image", 40, 1.4)}
  <div>No image yet</div>
  <small>Upload an image or pick an example above</small>
</div>
"""


def results_card_head(status_class, status_text):
    return f"""
<div class="ntc-card-head">
  <p class="ntc-card-title">AI Analysis Results</p>
  <span class="ntc-status {status_class}">{status_text}</span>
</div>
"""


def results_idle_html():
    return results_card_head("idle", "Waiting for image") + f"""
<div class="ntc-loading">
  <div class="ntc-orb">{icon("chip", 30, 1.5)}</div>
  <strong>Ready to analyse</strong>
  <span class="ntc-muted">Results appear here once you add an image.</span>
</div>
"""


def results_loading_html():
    return results_card_head("wait", "Analysing…") + """
<div class="ntc-loading">
  <div class="ntc-spinner"></div>
  <strong>Analysing image…</strong>
  <span class="ntc-muted">Running ConvNeXt Tiny on your image</span>
  <div class="ntc-shimmer"></div>
</div>
"""


def results_html(result):
    probs = result["probs"]
    top = max(range(len(probs)), key=lambda i: probs[i])
    label = CLASSES[top]
    pred_icon = icon("leaf", 40, 1.6, fill="currentColor") if label == "benign" else icon("alert", 40, 1.6)

    def bar(i):
        top_cls = ' class="top"' if i == top else ""
        return f"""
<div class="ntc-bar">
  <label>{CLASSES[i].capitalize()}</label>
  <div class="ntc-bar-row">
    <div class="ntc-track"><i class="{CLASSES[i]}" style="width:{probs[i] * 100:.1f}%"></i></div>
    <b{top_cls}>{probs[i]:.1%}</b>
  </div>
</div>"""

    return results_card_head("", "Analysis complete") + f"""
<div class="ntc-pred {label}">
  {pred_icon}
  <div class="ntc-pred-main"><small>Predicted Class</small><strong>{label.capitalize()}</strong></div>
  <div class="ntc-pred-conf"><small>Confidence</small><strong>{probs[top]:.1%}</strong></div>
</div>
{bar(0)}{bar(1)}
<div class="ntc-details">
  <header>{icon("info", 19)} Image Details</header>
  <table>
    <tr><td>Model</td><td>ConvNeXt Tiny</td></tr>
    <tr><td>Input image size</td><td>{IMG_SIZE} × {IMG_SIZE}</td></tr>
    <tr><td>Preprocessing</td><td>Resize, Normalise (ImageNet)</td></tr>
    <tr><td>Inference time</td><td>{result["seconds"]:.2f} seconds</td></tr>
    <tr><td>Date and time</td><td>{result["when"]}</td></tr>
  </table>
</div>
"""


def tips_html():
    tips = ["Use clear, well-lit images", "Ensure the lesion is in focus",
            "Avoid excessive blurring", "Include the full lesion in the frame"]
    items = "".join(f"<li>{icon('check', 19, 1.5)}<span>{t}</span></li>" for t in tips)
    return f"""
<div class="ntc-card ntc-tips"><h3>{icon("bulb", 24)} Tips for the best results</h3><ul>{items}</ul></div>
<div class="ntc-card ntc-warn">{icon("shield", 22)}
  <div>
    <strong>This is a research prototype and not a medical device.</strong>
    <p>The model's predictions are not medical diagnoses and should not replace professional medical advice.</p>
  </div>
</div>
"""


def feature(icon_name, title, text, small=False):
    size = 24 if small else 32
    return f"""
<div class="ntc-feature">
  <div class="ntc-orb{' sm' if small else ''}">{icon(icon_name, size)}</div>
  <div><h4>{title}</h4><p>{text}</p></div>
</div>"""


def footer_note(text):
    return f'<div class="ntc-card ntc-foot">{icon("info", 22)}<span>{text}</span></div>'


# =============================================================================
# State helpers
# =============================================================================
ss = st.session_state
ss.setdefault("page", "Analyse Image")
ss.setdefault("uploader_key", 0)
ss.setdefault("example", None)      # (path, label) when an example is chosen
ss.setdefault("rotation", 0)
ss.setdefault("example_seed", 7)
ss.setdefault("results", {})        # cache: image key -> result dict


def go(page):
    ss.page = page


def choose_example(path, label):
    ss.example = (path, label)
    ss.rotation = 0
    ss.uploader_key += 1  # empties the uploader so the example takes over


def clear_image():
    ss.example = None
    ss.rotation = 0
    ss.uploader_key += 1


def rotate_image():
    ss.rotation = (ss.rotation + 90) % 360


@st.dialog("Image preview", width="large")
def zoom_dialog(image, name):
    st.image(image, caption=name, width="stretch")


# =============================================================================
# Scan history (stored in the database, per user)
# =============================================================================
def save_scan(image, name, probs):
    """Store a finished scan for the logged-in user. Returns the case ID."""
    thumb = image.copy()
    thumb.thumbnail((512, 512))  # re-encoding also strips EXIF/location data
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    top = max(range(len(probs)), key=lambda i: probs[i])
    return db.save_lesion_case_and_prediction(
        user_id=ss["user_id"],
        lesion_tag=name or "Untitled scan",
        anatomical_site="Not specified",
        patient_age_bracket="Not specified",
        evolving_flag=False,
        bleeding_itching_flag=False,
        clinical_notes="",
        sanitized_image_bytes=buf.getvalue(),
        image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        benign_prob=probs[0],
        malignant_prob=probs[1],
        threshold_used=0.5,
        predicted_class=CLASSES[top],
    )


def load_history(user_id):
    """All of a user's scans, newest first, with images decrypted."""
    with db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.case_id, c.lesion_tag, c.created_at, c.encrypted_image_blob,
                   p.benign_prob, p.malignant_prob, p.predicted_class
            FROM lesion_cases c JOIN predictions p ON p.case_id = c.case_id
            WHERE c.user_id = ?
            ORDER BY c.created_at DESC
            """,
            (user_id,),
        ).fetchall()
    history = []
    for r in rows:
        img_uri = ""
        if r["encrypted_image_blob"]:
            try:
                jpeg = db.cipher.decrypt(r["encrypted_image_blob"])
                img_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            except Exception:
                pass  # e.g. saved with a different encryption key
        history.append({
            "case_id": r["case_id"],
            "name": r["lesion_tag"],
            "when": datetime.fromisoformat(r["created_at"]).astimezone(),  # UTC -> local time
            "img": img_uri,
            "benign": r["benign_prob"],
            "malignant": r["malignant_prob"],
            "label": r["predicted_class"],
        })
    return history


def history_row_html(h):
    img = (f'<img src="{h["img"]}" alt="Scan image">' if h["img"]
           else f'<div class="ntc-hist-noimg">{icon("image", 26, 1.4)}</div>')
    bars = "".join(
        f'<div class="ntc-hist-bar"><span>{cls.capitalize()}</span>'
        f'<div class="ntc-track"><i class="{cls}" style="width:{h[cls] * 100:.1f}%"></i></div>'
        f'<b>{h[cls]:.1%}</b></div>'
        for cls in CLASSES
    )
    return f"""
<div class="ntc-card ntc-hist-row">
  <div class="ntc-hist-img">{img}</div>
  <div class="ntc-hist-meta">
    <div class="ntc-hist-date">{h["when"].strftime("%d %b %Y")}</div>
    <div class="ntc-hist-time">{h["when"].strftime("%H:%M")}</div>
    <div class="ntc-muted ntc-hist-name">{escape(h["name"])}</div>
    <span class="ntc-badge {h["label"]}">{h["label"].capitalize()}</span>
  </div>
  <div class="ntc-hist-bars">{bars}</div>
</div>"""


def user_bar():
    """Bar across the top of every page: who's signed in, plus a log-out button."""
    username = ss.get("username", "")
    with st.container(key="userbar"):
        left, right = st.columns([5, 1], vertical_alignment="center")
        left.html(
            f'<div class="ntc-user"><span class="ntc-avatar">{escape(username[:1].upper())}</span>'
            f'<span>Signed in as <b>{escape(username)}</b></span></div>'
        )
        if right.button("Log out", icon=":material/logout:", type="tertiary", width="stretch"):
            # forget this user's on-screen state before signing out
            for k in ("results", "example", "page"):
                ss.pop(k, None)
            logout()


# =============================================================================
# Pages
# =============================================================================
def page_analyse():
    st.html(page_header(
        "Skin lesion classifier",
        "Analyse a Skin Lesion Image",
        "Upload a dermatoscopic or clinical image to get an AI-assisted classification.",
    ))

    # ---- row 1: uploader + examples ----
    top_left, top_right = st.columns([1.35, 1], gap="medium")
    with top_left:
        with st.container(key="card_drop"):
            uploaded = st.file_uploader(
                "Upload an image", type=["jpg", "jpeg", "png"],
                key=f"uploader_{ss.uploader_key}", label_visibility="collapsed",
            )

    with top_right:
        with st.container(key="card_examples"):
            head_l, head_r = st.columns([3, 1.3], vertical_alignment="center")
            head_l.html('<p class="ntc-card-title">Example Images</p>')
            with head_r:
                with st.container(key="view_more"):
                    if st.button("View more", icon=":material/arrow_forward:", type="tertiary"):
                        ss.example_seed += 1
                        st.rerun()
            examples = pick_examples(ss.example_seed)
            if examples:
                for col, (path, label) in zip(st.columns(4, gap="small"), examples):
                    with col:
                        st.image(square_thumb(path), width="stretch")
                        st.button(label, key=f"ex_{path}", type="tertiary",
                                  on_click=choose_example, args=(path, label))
            else:
                st.html('<p class="ntc-muted">Example images need the data/val folder.</p>')

    # ---- work out which image we're showing ----
    image, name, source = None, "", None
    if uploaded is not None:
        ss.example = None
        image = Image.open(uploaded).convert("RGB")
        name, source = uploaded.name, f"upload:{uploaded.file_id}"
    elif ss.example:
        path, label = ss.example
        image = Image.open(path).convert("RGB")
        name, source = f"{Path(path).name} · example", f"example:{path}"
    if image is not None and ss.rotation:
        image = image.rotate(-ss.rotation, expand=True)

    # ---- row 2: image, results, tips ----
    col_img, col_res, col_tips = st.columns([1, 1, 0.8], gap="medium")

    with col_img:
        with st.container(key="card_image"):
            st.html(f'<div class="ntc-card-head"><p class="ntc-card-title">Uploaded Image</p>'
                    f'<span class="ntc-muted">{name}</span></div>')
            if image is None:
                st.html(empty_image_html())
            else:
                st.image(image, width="stretch")
            with st.container(key="img_actions"):
                a1, a2, a3 = st.columns(3)
                no_img = image is None
                if a1.button("Zoom", icon=":material/zoom_in:", type="tertiary", disabled=no_img):
                    zoom_dialog(image, name)
                a2.button("Rotate", icon=":material/rotate_right:", type="tertiary",
                          disabled=no_img, on_click=rotate_image)
                a3.button("Remove", icon=":material/delete:", type="tertiary",
                          disabled=no_img, on_click=clear_image)

    with col_tips:
        st.html(tips_html())

    with col_res:
        with st.container(key="card_results"):
            slot = st.empty()
            if image is None:
                slot.html(results_idle_html())
                return
            if not MODEL_PATH.exists():
                slot.html(results_card_head("idle", "Model missing"))
                st.error(f"Model weights not found at {MODEL_PATH}")
                return

            key = f"{source}|{ss.rotation}"
            if key not in ss.results:
                slot.html(results_loading_html())
                started = time.time()
                try:
                    probs, seconds = predict(image, load_model())
                except Exception as e:
                    slot.empty()
                    st.error(f"Unable to analyse image: {e}")
                    return
                time.sleep(max(0.0, MIN_LOADING_SECONDS - (time.time() - started)))
                ss.results[key] = {
                    "probs": probs,
                    "seconds": seconds,
                    "when": datetime.now().strftime("%d %b %Y, %H:%M"),
                }
                try:
                    save_scan(image, name, probs)  # shows up on the Past Results page
                except Exception as e:
                    st.warning(f"Result shown but not saved to your history: {e}")
            slot.html(results_html(ss.results[key]))


def page_history():
    st.html(page_header(
        "Activity",
        "Past <em>Results</em>",
        "Every scan you've made, newest first.",
    ))
    history = load_history(ss["user_id"])
    if not history:
        st.html(f"""
<div class="ntc-card ntc-loading" style="min-height:16rem">
  <div class="ntc-orb">{icon("chart", 30, 1.5)}</div>
  <strong>No scans yet</strong>
  <span class="ntc-muted">Analyse an image and it will appear here.</span>
</div>""")
        return

    n_mal = sum(h["label"] == "malignant" for h in history)
    stats = [
        ("Total scans", len(history)),
        ("Benign", len(history) - n_mal),
        ("Malignant", n_mal),
        ("Last scan", history[0]["when"].strftime("%d %b, %H:%M")),
    ]
    st.html('<div class="ntc-stats">' + "".join(
        f'<div class="ntc-card"><small>{k}</small><strong>{v}</strong></div>' for k, v in stats
    ) + "</div>")
    st.html('<div class="ntc-hist">' + "".join(history_row_html(h) for h in history) + "</div>")


def page_about():
    st.html(page_header(
        "About",
        "NTC — <em>NAH</em> to Cancer",
        "A student-led initiative using artificial intelligence to support early skin cancer detection.",
    ))
    st.html(f"""
<div class="ntc-card" style="margin-top:.6rem">
  <div class="ntc-cause">
    <div class="ntc-body">
      <h2 class="ntc-h2">Our Cause</h2>
      <p>Skin cancer is one of the most common cancers worldwide, but it is also one of the most preventable when detected early.</p>
      <p>At NTC — NAH to Cancer, we believe that technology can help make early detection more accessible, understandable, and impactful. Our project combines artificial intelligence with medical imaging to support individuals and healthcare professionals in identifying potentially concerning skin lesions.</p>
      <p>We are a team of four students united by a shared goal: to use our skills to make a positive difference in healthcare.</p>
      <p>Because a future with fewer skin cancer cases is a future worth building.</p>
    </div>
    <div class="ntc-vrule"></div>
    <div>
      {feature("target", "Our Mission", "To explore responsible AI technologies that could support earlier recognition of concerning skin lesions and contribute to more informed healthcare.")}
      {feature("eye", "Our Vision", "A future where technology, education, and medical expertise work together to make skin health awareness and timely care more accessible.")}
      {feature("users", "Our Team", "NTC — NAH to Cancer (N A H)<br>A team of four students passionate about healthcare, innovation, and creating real-world impact.")}
    </div>
  </div>
</div>
<div class="ntc-grid two">
  <div class="ntc-card ntc-body">
    <h2 class="ntc-h2">Why “NAH to Cancer”?</h2>
    <p>Our name represents both our team’s initials — N, A, H — and our shared determination to challenge cancer through curiosity, collaboration, and innovation.</p>
    <p>We may be starting with a student-led project, but our ambition is bigger than a single application. We want to raise awareness, demonstrate the potential of AI in healthcare, and contribute to a future where fewer people face the life-changing impact of skin cancer.</p>
    <div class="ntc-motto"><i>{icon("heart", 16, 0, fill="currentColor")}</i><span>Four minds. One mission. <b>NAH to Cancer.</b></span></div>
  </div>
  <div class="ntc-card ntc-values">
    <h2 class="ntc-h2" style="padding-bottom:0">Our Values</h2>
    <style>.ntc-values .ntc-h2::after{{display:none}}</style>
    {feature("bulb", "Impact", "Using innovation for a healthier, brighter future.", small=True)}
    {feature("shield", "Responsibility", "Developing and communicating AI ethically and transparently.", small=True)}
    {feature("users", "Collaboration", "Combining diverse skills and perspectives to create meaningful change.", small=True)}
    {feature("leaf", "Awareness", "Empowering individuals with knowledge about skin health.", small=True)}
  </div>
</div>
{footer_note("NTC is currently a research and educational prototype. It is not a clinically validated diagnostic tool and should not be used to make medical decisions.")}
""")


def page_model():
    st.html(page_header(
        "Model information",
        "How the <em>model</em> works",
        "A ConvNeXt Tiny network, fine-tuned to sort skin lesion images into benign or malignant.",
    ))
    st.html(f"""
<div class="ntc-grid two">
  <div class="ntc-card">
    <h2 class="ntc-h2">Architecture</h2>
    {feature("layers", "ConvNeXt Tiny", "A modern convolutional network (via the timm library), pretrained on ImageNet and fine-tuned for two classes.", small=True)}
    {feature("image", "Input", f"RGB image resized to {IMG_SIZE} × {IMG_SIZE} pixels and normalised with ImageNet mean and standard deviation.", small=True)}
    {feature("chart", "Output", "Softmax scores for <b>benign</b> and <b>malignant</b>. The highest score is the predicted class.", small=True)}
  </div>
  <div class="ntc-card">
    <h2 class="ntc-h2">Training setup</h2>
    <table class="ntc-kv">
      <tr><td>Epochs</td><td>10</td></tr>
      <tr><td>Batch size</td><td>32</td></tr>
      <tr><td>Optimiser</td><td>AdamW · learning rate 1e-4 · weight decay 0.05</td></tr>
      <tr><td>Schedule</td><td>Cosine annealing</td></tr>
      <tr><td>Loss</td><td>Cross-entropy, weighted for class imbalance</td></tr>
      <tr><td>Augmentation</td><td>Random resized crop, horizontal and vertical flips, rotation ±30°, colour jitter</td></tr>
      <tr><td>Checkpoint</td><td>Best validation ROC AUC is kept</td></tr>
      <tr><td>Running on</td><td>{device.upper()}</td></tr>
    </table>
  </div>
</div>
<div class="ntc-card" style="margin-top:1.1rem">
  <h2 class="ntc-h2">Reading the results</h2>
  {feature("sliders", "Scores are not probabilities of disease", "The bars show the model's softmax scores. They are not calibrated, so 87% does not mean an 87% chance of cancer.", small=True)}
  {feature("target", "Evaluation metrics", "The model is judged on validation ROC AUC and sensitivity (recall on malignant cases), because missing a malignant lesion matters most.", small=True)}
</div>
{footer_note("Like any model, it can be wrong — especially on image types it did not see in training. It is not a substitute for a dermatologist.")}
""")


def page_usage():
    st.html(page_header(
        "Usage guide",
        "Getting the <em>best</em> results",
        "Four steps from image to result, plus what to keep in mind.",
    ))
    st.html(f"""
<div class="ntc-grid two">
  <div class="ntc-card">
    <h2 class="ntc-h2">How to use it</h2>
    <ol class="ntc-steps">
      <li><span><b>Open Analyse Image</b> from the menu on the left.</span></li>
      <li><span><b>Add an image</b>: drag and drop a JPG or PNG, click Browse Files, or pick one of the example images.</span></li>
      <li><span><b>Wait a moment</b> while the model analyses it. The results panel shows a loading state.</span></li>
      <li><span><b>Read the results</b>: the predicted class, confidence, and the score for each class. Use Rotate, Zoom or Remove to adjust.</span></li>
    </ol>
  </div>
  <div class="ntc-card ntc-tips">
    <h3>{icon("bulb", 24)} Image tips</h3>
    <ul>
      <li>{icon("check", 19, 1.5)}<span>Use clear, well-lit images taken close to the skin</span></li>
      <li>{icon("check", 19, 1.5)}<span>Keep the lesion in focus and in the centre</span></li>
      <li>{icon("check", 19, 1.5)}<span>Include the whole lesion in the frame</span></li>
      <li>{icon("x", 19, 1.5)}<span>Avoid blur, heavy shadows, filters or hair covering the lesion</span></li>
      <li>{icon("x", 19, 1.5)}<span>Don't upload images that aren't of skin — the model will still give an answer, but it means nothing</span></li>
    </ul>
  </div>
</div>
{footer_note("If you're worried about a mole or skin change, see a doctor or dermatologist, whatever this app says.")}
""")


# =============================================================================
# App
# =============================================================================

# Initialise database
db.init_db()

# Configure Streamlit
st.set_page_config(page_title="NTC — NAH to Cancer", page_icon="🎗️", layout="wide")

# Require login
if not render_login():
    st.stop()

# Load main interface
st.html(CSS)

PAGES = {
    "Analyse Image": (":material/upload:", page_analyse),
    "Past Results": (":material/history:", page_history),
    "About": (":material/info:", page_about),
    "Model Information": (":material/bar_chart:", page_model),
    "Usage Guide": (":material/description:", page_usage),
}

with st.sidebar:
    st.html(sidebar_logo())
    for page_name, (page_icon, _) in PAGES.items():
        st.button(
            page_name, icon=page_icon, width="stretch", key=f"nav_{page_name}",
            type="primary" if ss.page == page_name else "tertiary",
            on_click=go, args=(page_name,),
        )
    st.html(sidebar_footer())

user_bar()
PAGES[ss.page][1]()
