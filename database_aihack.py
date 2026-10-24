# database.py
import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from cryptography.fernet import Fernet

DB_PATH = "melanoma_vault.db"
KEY_FILE = "vault_master.key"


def load_or_create_encryption_key() -> bytes:
    """
    Loads the symmetric AES Fernet key from environment variable or local key file.
    In production, inject VAULT_ENCRYPTION_KEY via environment variables / secret manager.
    """
    env_key = os.environ.get("VAULT_ENCRYPTION_KEY")
    if env_key:
        return env_key.encode("utf-8")
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read().strip()
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    return key


cipher = Fernet(load_or_create_encryption_key())


def get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with foreign key enforcement and row dictionary mapping."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db() -> None:
    """Creates the users, lesion_cases, predictions, and audit_logs tables."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'patient',
                consent_accepted INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lesion_cases (
                case_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                lesion_tag TEXT NOT NULL,
                anatomical_site TEXT NOT NULL,
                patient_age_bracket TEXT NOT NULL,
                evolving_flag INTEGER NOT NULL DEFAULT 0,
                bleeding_itching_flag INTEGER NOT NULL DEFAULT 0,
                encrypted_notes BLOB,
                encrypted_image_blob BLOB,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS predictions (
                pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL UNIQUE,
                image_sha256 TEXT NOT NULL,
                model_name TEXT NOT NULL DEFAULT 'convnext_tiny',
                benign_prob REAL NOT NULL,
                malignant_prob REAL NOT NULL,
                threshold_used REAL NOT NULL,
                predicted_class TEXT NOT NULL,
                clinician_verdict TEXT NOT NULL DEFAULT 'pending_review',
                created_at TEXT NOT NULL,
                FOREIGN KEY (case_id) REFERENCES lesion_cases(case_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                case_id TEXT,
                action_type TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_cases_user_lesion
                ON lesion_cases(user_id, lesion_tag);
            """
        )


# ---- 1. User Authentication & Registration ----

def _hash_password(password: str, salt_hex: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        200_000,
    )
    return dk.hex()


def register_user(username: str, password: str, role: str = "patient") -> str:
    """Creates a new pseudonymized user account and returns the generated user_id."""
    user_id = f"USR-{secrets.token_hex(6).upper()}"
    salt_hex = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt_hex)
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, username, password_hash, password_salt, role, consent_accepted, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (user_id, username.strip().lower(), pw_hash, salt_hex, role, now),
        )
        conn.execute(
            "INSERT INTO audit_logs (user_id, case_id, action_type, timestamp_utc) VALUES (?, NULL, 'USER_REGISTERED', ?)",
            (user_id, now),
        )
    return user_id


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verifies user credentials and returns user metadata if valid."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip().lower(),)
        ).fetchone()
        if not row:
            return None
        expected_hash = _hash_password(password, row["password_salt"])
        if secrets.compare_digest(expected_hash, row["password_hash"]):
            return {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}
    return None


# ---- 2. Storing User Input, Encrypted Image & Model Prediction ----

def save_lesion_case_and_prediction(
    user_id: str,
    lesion_tag: str,
    anatomical_site: str,
    patient_age_bracket: str,
    evolving_flag: bool,
    bleeding_itching_flag: bool,
    clinical_notes: str,
    sanitized_image_bytes: Optional[bytes],
    image_sha256: str,
    benign_prob: float,
    malignant_prob: float,
    threshold_used: float,
    predicted_class: str,
) -> str:
    """
    Encrypts sensitive clinical notes and optional EXIF-stripped image bytes via AES (Fernet),
    and commits both the user input and convnext_tiny prediction in a single atomic transaction.
    """
    case_id = f"MEL-{secrets.token_hex(6).upper()}"
    now = datetime.now(timezone.utc).isoformat()

    enc_notes = cipher.encrypt(clinical_notes.encode("utf-8")) if clinical_notes else None
    enc_img = cipher.encrypt(sanitized_image_bytes) if sanitized_image_bytes else None

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO lesion_cases (
                case_id, user_id, lesion_tag, anatomical_site, patient_age_bracket,
                evolving_flag, bleeding_itching_flag, encrypted_notes, encrypted_image_blob, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                user_id,
                lesion_tag.strip(),
                anatomical_site.strip(),
                patient_age_bracket,
                int(evolving_flag),
                int(bleeding_itching_flag),
                enc_notes,
                enc_img,
                now,
            ),
        )

        conn.execute(
            """
            INSERT INTO predictions (
                case_id, image_sha256, model_name, benign_prob, malignant_prob,
                threshold_used, predicted_class, created_at
            ) VALUES (?, ?, 'convnext_tiny', ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                image_sha256,
                round(benign_prob, 4),
                round(malignant_prob, 4),
                round(threshold_used, 4),
                predicted_class,
                now,
            ),
        )

        conn.execute(
            """
            INSERT INTO audit_logs (user_id, case_id, action_type, timestamp_utc)
            VALUES (?, ?, 'CASE_AND_PREDICTION_STORED', ?)
            """,
            (user_id, case_id, now),
        )

    return case_id


# ---- 3. Querying User History & Longitudinal Lesion Progression ----

def get_user_lesion_history(user_id: str) -> List[Dict[str, Any]]:
    """
    Uses a SQL WITH clause (CTE) and window function (LAG) to retrieve a user's stored
    lesion scans alongside the malignancy probability delta from the previous scan of the same mole.
    """
    query = """
    WITH LongitudinalScans AS (
        SELECT
            c.case_id,
            c.lesion_tag,
            c.anatomical_site,
            c.patient_age_bracket,
            c.evolving_flag,
            c.bleeding_itching_flag,
            c.encrypted_notes,
            c.created_at,
            p.benign_prob,
            p.malignant_prob,
            p.threshold_used,
            p.predicted_class,
            p.clinician_verdict,
            LAG(p.malignant_prob) OVER (
                PARTITION BY c.user_id, c.lesion_tag
                ORDER BY c.created_at ASC
            ) AS previous_malignant_prob
        FROM lesion_cases c
        INNER JOIN predictions p ON c.case_id = p.case_id
        WHERE c.user_id = ?
    )
    SELECT * FROM LongitudinalScans
    ORDER BY created_at DESC;
    """
    records = []
    with get_connection() as conn:
        rows = conn.execute(query, (user_id,)).fetchall()
        for r in rows:
            decrypted_note = (
                cipher.decrypt(r["encrypted_notes"]).decode("utf-8")
                if r["encrypted_notes"]
                else ""
            )
            prev_prob = r["previous_malignant_prob"]
            prob_delta = (
                round(r["malignant_prob"] - prev_prob, 4)
                if prev_prob is not None
                else None
            )
            records.append({
                "case_id": r["case_id"],
                "lesion_tag": r["lesion_tag"],
                "anatomical_site": r["anatomical_site"],
                "patient_age_bracket": r["patient_age_bracket"],
                "evolving_flag": bool(r["evolving_flag"]),
                "bleeding_itching_flag": bool(r["bleeding_itching_flag"]),
                "clinical_notes": decrypted_note,
                "benign_prob": r["benign_prob"],
                "malignant_prob": r["malignant_prob"],
                "malignant_prob_delta_since_last_scan": prob_delta,
                "predicted_class": r["predicted_class"],
                "clinician_verdict": r["clinician_verdict"],
                "created_at": r["created_at"],
            })
    return records


# ---- 4. Clinician Feedback & GDPR Right to Erasure ----

def update_clinician_verdict(user_id: str, case_id: str, verdict: str) -> bool:
    """Allows a clinician to record agreement or override ('concur', 'override_benign', 'biopsy_ordered')."""
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE predictions SET clinician_verdict = ? WHERE case_id = ?",
            (verdict, case_id),
        )
        if cur.rowcount > 0:
            conn.execute(
                "INSERT INTO audit_logs (user_id, case_id, action_type, timestamp_utc) VALUES (?, ?, 'CLINICIAN_VERDICT_UPDATED', ?)",
                (user_id, case_id, now),
            )
            return True
    return False


def delete_case_gdpr(user_id: str, case_id: str) -> bool:
    """
    Permanently deletes a stored lesion case and its prediction (via ON DELETE CASCADE)
    to fulfill a GDPR Right-to-Erasure request, recording only the deletion event in the audit log.
    """
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM lesion_cases WHERE case_id = ? AND user_id = ?",
            (case_id, user_id),
        )
        if cur.rowcount > 0:
            conn.execute(
                "INSERT INTO audit_logs (user_id, case_id, action_type, timestamp_utc) VALUES (?, ?, 'GDPR_CASE_DELETED', ?)",
                (user_id, case_id, now),
            )
            return True
    return False
