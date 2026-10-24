import os
import hashlib
import secrets
from io import BytesIO
from pathlib import Path

from cryptography.fernet import Fernet
from PIL import Image

# Set the test key BEFORE importing the database module.
os.environ["VAULT_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

import database_aihack as db

# Keep dummy records separate from your real database.
db.DB_PATH = "test_melanoma_vault.db"
db.init_db()

# 1. Create a dummy account.
username = f"test_patient_{secrets.token_hex(4)}"

user_id = db.register_user(
    username=username,
    password="DummyPassword123!",
)

# 2. Check that login works.
user = db.authenticate_user(username, "DummyPassword123!")
assert user is not None
assert user["user_id"] == user_id
assert db.authenticate_user(username, "wrong-password") is None

# 3. Read the test image and re-encode without its original metadata.
image_path = Path(__file__).parent / "6299.jpg"

with Image.open(image_path) as original:
    rgb = original.convert("RGB")
    clean_image = Image.new("RGB", rgb.size)
    clean_image.paste(rgb)

    buffer = BytesIO()
    clean_image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

# 4. Save dummy input and a FAKE model result.
case_id = db.save_lesion_case_and_prediction(
    user_id=user_id,
    lesion_tag="TEST-MOLE-001",
    anatomical_site="left forearm",
    patient_age_bracket="20-29",
    evolving_flag=False,
    bleeding_itching_flag=False,
    clinical_notes="Synthetic test record. Not a clinical assessment.",
    sanitized_image_bytes=image_bytes,
    image_sha256=hashlib.sha256(image_bytes).hexdigest(),
    benign_prob=0.80,
    malignant_prob=0.20,
    threshold_used=0.50,
    predicted_class="benign",
)

# 5. Retrieve the record and check the decrypted notes.
history = db.get_user_lesion_history(user_id)

assert len(history) == 1
assert history[0]["case_id"] == case_id
assert history[0]["clinical_notes"] == (
    "Synthetic test record. Not a clinical assessment."
)

# 6. Confirm the stored image is encrypted and decrypts correctly.
conn = db.get_connection()
try:
    row = conn.execute(
        "SELECT encrypted_image_blob FROM lesion_cases WHERE case_id = ?",
        (case_id,),
    ).fetchone()

    assert row["encrypted_image_blob"] != image_bytes
    assert db.cipher.decrypt(row["encrypted_image_blob"]) == image_bytes
finally:
    conn.close()

print("PASS: registration and login")
print("PASS: case and fake prediction saved")
print("PASS: history and decrypted notes retrieved")
print("PASS: image encryption round trip")

# 7. Remove the dummy case and account.
assert db.delete_case_gdpr(user_id, case_id)
assert db.get_user_lesion_history(user_id) == []

conn = db.get_connection()
try:
    with conn:
        conn.execute("DELETE FROM audit_logs WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
finally:
    conn.close()

print("PASS: dummy records deleted")