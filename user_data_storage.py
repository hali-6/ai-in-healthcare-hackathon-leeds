# user_data_storage.py

import sqlite3
import hashlib
import secrets
from datetime import datetime, timezone
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from database_aihack import get_connection, cipher


# ---------------------------------------
# 1. CREATE USER DATA TABLE
# ---------------------------------------

def init_user_data_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_scan_records (
                record_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                encrypted_name BLOB NOT NULL,
                encrypted_image BLOB NOT NULL,
                encrypted_filename BLOB,
                image_sha256 TEXT NOT NULL,
                encrypted_prediction BLOB NOT NULL,
                benign_probability REAL NOT NULL,
                malignant_probability REAL NOT NULL,
                scan_date TEXT NOT NULL,

                FOREIGN KEY (user_id)
                REFERENCES users(user_id)
                ON DELETE CASCADE
            )
        """)


# ---------------------------------------
# 2. ENCRYPT AND DECRYPT DATA
# ---------------------------------------

def encrypt_text(value):
    return cipher.encrypt(str(value).encode("utf-8"))


def decrypt_text(value):
    return cipher.decrypt(value).decode("utf-8")


# ---------------------------------------
# 3. VALIDATE AND SANITISE IMAGE
# ---------------------------------------

def prepare_image(image_bytes):
    if not image_bytes or len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError("Image must be between 1 byte and 10 MB")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("Only JPEG and PNG are allowed")

            if image.width * image.height > 20_000_000:
                raise ValueError("Image dimensions are too large")

            rgb_image = image.convert("RGB")

            # Re-encode to remove original image metadata
            output = BytesIO()
            rgb_image.save(output, format="PNG")

            return output.getvalue()

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image file") from exc


# ---------------------------------------
# 4. SAVE USER IMAGE AND AI RESULT
# ---------------------------------------

def save_user_scan(
    user_id,
    name,
    image_bytes,
    filename,
    predicted_class,
    benign_probability,
    malignant_probability
):
    if predicted_class not in ("benign", "malignant"):
        raise ValueError("Invalid prediction class")

    benign_probability = float(benign_probability)
    malignant_probability = float(malignant_probability)

    if not (
        0 <= benign_probability <= 1
        and 0 <= malignant_probability <= 1
        and abs(benign_probability + malignant_probability - 1) < 0.01
    ):
        raise ValueError("Invalid prediction probabilities")

    record_id = "SCAN-" + secrets.token_hex(8).upper()
    scan_date = datetime.now(timezone.utc).isoformat()

    clean_image = prepare_image(image_bytes)

    encrypted_name = encrypt_text(name)
    encrypted_image = cipher.encrypt(clean_image)
    encrypted_filename = encrypt_text(filename)
    encrypted_prediction = encrypt_text(predicted_class)

    image_sha256 = hashlib.sha256(clean_image).hexdigest()

    with get_connection() as conn:

        # Check that the user exists
        user = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if user is None:
            raise ValueError("User does not exist")

        conn.execute("""
            INSERT INTO user_scan_records (
                record_id,
                user_id,
                encrypted_name,
                encrypted_image,
                encrypted_filename,
                image_sha256,
                encrypted_prediction,
                benign_probability,
                malignant_probability,
                scan_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id,
            user_id,
            encrypted_name,
            encrypted_image,
            encrypted_filename,
            image_sha256,
            encrypted_prediction,
            benign_probability,
            malignant_probability,
            scan_date
        ))

    return record_id


# ---------------------------------------
# 5. RETRIEVE USER SCAN HISTORY
# ---------------------------------------

def get_user_scans(user_id):

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT *
            FROM user_scan_records
            WHERE user_id = ?
            ORDER BY scan_date DESC
        """, (user_id,)).fetchall()

    results = []

    for row in rows:
        results.append({
            "record_id": row["record_id"],
            "user_id": row["user_id"],
            "name": decrypt_text(row["encrypted_name"]),
            "filename": decrypt_text(row["encrypted_filename"]),
            "predicted_class": decrypt_text(
                row["encrypted_prediction"]
            ),
            "benign_probability": row["benign_probability"],
            "malignant_probability": row["malignant_probability"],
            "scan_date": row["scan_date"]
        })

    return results


# ---------------------------------------
# 6. RETRIEVE ENCRYPTED IMAGE
# ---------------------------------------

def get_user_image(user_id, record_id):

    with get_connection() as conn:
        row = conn.execute("""
            SELECT encrypted_image
            FROM user_scan_records
            WHERE user_id = ? AND record_id = ?
        """, (user_id, record_id)).fetchone()

    if row is None:
        return None

    return cipher.decrypt(row["encrypted_image"])


# ---------------------------------------
# 7. DELETE USER SCAN
# ---------------------------------------

def delete_user_scan(user_id, record_id):

    with get_connection() as conn:
        cursor = conn.execute("""
            DELETE FROM user_scan_records
            WHERE user_id = ? AND record_id = ?
        """, (user_id, record_id))

        return cursor.rowcount > 0