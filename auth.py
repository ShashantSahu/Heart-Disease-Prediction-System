"""
CardioScore Pro - accounts and per-user assessment history (EXTRA FEATURE C + D).

* Passwords are never stored. Only a salted PBKDF2-HMAC-SHA256 hash is saved
  (Python standard library, 600,000 iterations, random 16-byte salt per user).
* SQLite database (data/cardioscore.db) with parameterised queries only.
* EVERY history query is filtered by user_id, so one user can never read,
  export or delete another user's assessments.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.environ.get(
    "CARDIOSCORE_DB", Path(__file__).resolve().parent / "data" / "cardioscore.db"))
ALGORITHM = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 600_000
USERNAME_RE = re.compile(r"^[a-z0-9_.]{3,30}$")
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
MIN_PASSWORD_LENGTH = 8
_DUMMY_HASH = None


class AuthError(ValueError):
    """Raised for invalid sign-up data or a duplicate username."""


@contextmanager
def _connect():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                email         TEXT,
                full_name     TEXT NOT NULL DEFAULT '',
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assessments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at   TEXT NOT NULL,
                model_name   TEXT NOT NULL,
                probability  REAL NOT NULL,
                risk_level   TEXT NOT NULL,
                inputs_json  TEXT NOT NULL,
                factors_json TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_assessments_user
                ON assessments (user_id, created_at);
        """)
        # Safe migration for databases created by older username-only versions.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "email" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL")


# ------------------------------------------------------------------ passwords
def hash_password(password: str, iterations: int | None = None) -> str:
    iterations = iterations or PBKDF2_ITERATIONS
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != ALGORITHM:
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                     bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, AttributeError):
        return False


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def validate_new_account(username: str, password: str) -> None:
    if not USERNAME_RE.match(normalize_username(username)):
        raise AuthError("Username must be 3-30 characters: letters, digits, '_' or '.'.")
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        raise AuthError("Password must contain at least one letter and one digit.")
    if normalize_username(username) == password.lower():
        raise AuthError("Password must not be the same as the username.")


# ---------------------------------------------------------------------- users
def create_user(username: str, password: str, full_name: str = "") -> dict:
    validate_new_account(username, password)
    username = normalize_username(username)
    full_name = (full_name or "").strip()[:60]
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, full_name, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, full_name, hash_password(password), _now()))
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        raise AuthError("This username is already taken.") from None
    return {"id": user_id, "username": username, "full_name": full_name}


def authenticate(username: str, password: str):
    """Return the user dict if the credentials are right, otherwise None."""
    global _DUMMY_HASH
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?",
                           (normalize_username(username),)).fetchone()
    if row is None:
        # Spend the same time as a real check so usernames cannot be probed by timing.
        if _DUMMY_HASH is None:
            _DUMMY_HASH = hash_password("dummy-password-1")
        verify_password(password or "", _DUMMY_HASH)
        return None
    if verify_password(password or "", row["password_hash"]):
        return {"id": row["id"], "username": row["username"], "full_name": row["full_name"]}
    return None


# ---------------------------------------------------------- email accounts
def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def create_email_user(email: str, password: str, full_name: str = "") -> dict:
    """Create an account using email; legacy username accounts remain supported."""
    email = normalize_email(email)
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise AuthError("Enter a valid email address.")
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        raise AuthError("Password must contain at least one letter and one digit.")
    full_name = (full_name or "").strip()[:60]
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, email, full_name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (email, email, full_name, hash_password(password), _now()))
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        raise AuthError("An account with this email already exists. Please log in.") from None
    return {"id": user_id, "username": email, "email": email, "full_name": full_name}


def authenticate_email(email: str, password: str):
    """Authenticate by email. Username fallback preserves access for older accounts."""
    global _DUMMY_HASH
    email = normalize_email(email)
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE lower(email) = ? OR (email IS NULL AND lower(username) = ?)",
                           (email, email)).fetchone()
    if row is None:
        if _DUMMY_HASH is None:
            _DUMMY_HASH = hash_password("dummy-password-1")
        verify_password(password or "", _DUMMY_HASH)
        return None
    if verify_password(password or "", row["password_hash"]):
        return {"id": row["id"], "username": row["username"],
                "email": row["email"] or "", "full_name": row["full_name"]}
    return None


# ---------------------------------------------------------------- assessments
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _row_to_assessment(row) -> dict:
    return {
        "id": row["id"], "user_id": row["user_id"], "created_at": row["created_at"],
        "model_name": row["model_name"], "probability": row["probability"],
        "risk_level": row["risk_level"], "inputs": json.loads(row["inputs_json"]),
        "factors": json.loads(row["factors_json"]),
    }


def save_assessment(user_id: int, model_name: str, probability: float, risk_level: str,
                    inputs: dict, factors: list | None = None) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO assessments (user_id, created_at, model_name, probability, "
            "risk_level, inputs_json, factors_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, _now(), model_name, float(probability), risk_level,
             json.dumps(inputs), json.dumps(factors or [])))
        return cur.lastrowid


def list_assessments(user_id: int) -> list:
    """All assessments of THIS user, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM assessments WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (user_id,)).fetchall()
    return [_row_to_assessment(r) for r in rows]


def get_assessment(user_id: int, assessment_id: int):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM assessments WHERE id = ? AND user_id = ?",
                           (assessment_id, user_id)).fetchone()
    return _row_to_assessment(row) if row else None


def delete_assessment(user_id: int, assessment_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM assessments WHERE id = ? AND user_id = ?",
                           (assessment_id, user_id))
        return cur.rowcount > 0
