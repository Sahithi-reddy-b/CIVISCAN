"""Authentication, profile management, and officer administration services."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import smtplib
import sqlite3
import ssl
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from config import (
    AUTH_LOCKOUT_MINUTES,
    AUTH_MAX_LOGIN_ATTEMPTS,
    PASSWORD_RESET_PUBLIC_URL,
    PASSWORD_RESET_TTL_MINUTES,
    SMTP_FROM_EMAIL,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USERNAME,
)
from database import DATABASE_PATH


PASSWORD_ITERATIONS = 200_000
SALT_BYTES = 16
ROLE_CITIZEN = "Citizen"
ROLE_MUNICIPAL_OFFICER = "Municipal Officer"
VALID_ROLES = {ROLE_CITIZEN, ROLE_MUNICIPAL_OFFICER}
ACTIVE_STATUS = "active"
INACTIVE_STATUS = "inactive"
_PASSWORD_HASHER = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_PHONE_PATTERN = re.compile(r"^\+?[0-9][0-9 .()\-]{5,19}$")

USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('Citizen', 'Municipal Officer')),
    email TEXT,
    phone_number TEXT,
    password_hash TEXT,
    employee_id TEXT,
    department TEXT,
    designation TEXT,
    jurisdiction TEXT,
    account_status TEXT NOT NULL DEFAULT 'active',
    is_admin INTEGER NOT NULL DEFAULT 0,
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until INTEGER NOT NULL DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
)
"""

RESET_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    token_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at INTEGER NOT NULL,
    used_at INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

LOGIN_ATTEMPTS_SQL = """
CREATE TABLE IF NOT EXISTS login_attempts (
    identifier_hash TEXT PRIMARY KEY,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class _ClosingConnection(sqlite3.Connection):
    """Close SQLite handles after their transaction context exits."""

    def __exit__(self, exception_type, exception, traceback):
        try:
            return super().__exit__(exception_type, exception, traceback)
        finally:
            self.close()


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(
        Path(DATABASE_PATH),
        timeout=30,
        factory=_ClosingConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_users_table() -> None:
    """Create or safely extend the user schema without replacing existing rows."""
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(USERS_TABLE_SQL)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(users)")}
        additions = {
            "email": "TEXT",
            "phone_number": "TEXT",
            "password_hash": "TEXT",
            "employee_id": "TEXT",
            "department": "TEXT",
            "designation": "TEXT",
            "jurisdiction": "TEXT",
            "account_status": "TEXT NOT NULL DEFAULT 'active'",
            "is_admin": "INTEGER NOT NULL DEFAULT 0",
            "failed_login_attempts": "INTEGER NOT NULL DEFAULT 0",
            "locked_until": "INTEGER NOT NULL DEFAULT 0",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        }
        for column, definition in additions.items():
            if column not in columns:
                connection.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")
        connection.execute(
            "UPDATE users SET password_hash = password "
            "WHERE password_hash IS NULL AND password IS NOT NULL"
        )
        connection.execute(
            "UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
        )
        connection.execute(
            "UPDATE users SET updated_at = COALESCE(created_at, CURRENT_TIMESTAMP) "
            "WHERE updated_at IS NULL"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique "
            "ON users(email COLLATE NOCASE) WHERE email IS NOT NULL AND email <> ''"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_employee_unique "
            "ON users(employee_id COLLATE NOCASE) WHERE employee_id IS NOT NULL AND employee_id <> ''"
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_users_role_status ON users(role, account_status)")
        connection.execute(RESET_TABLE_SQL)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_reset_user ON password_reset_tokens(user_id)")
        connection.execute(LOGIN_ATTEMPTS_SQL)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _normalise_email(email: str) -> str:
    value = email.strip().lower()
    if len(value) > 254 or not _EMAIL_PATTERN.fullmatch(value):
        raise ValueError("Enter a valid email address.")
    return value


def _validate_password(password: str) -> None:
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters long.")
    if len(password) > 1024 or not password.strip():
        raise ValueError("Password is invalid.")


def _validate_name(name: str) -> str:
    value = " ".join(name.split())
    if not 2 <= len(value) <= 100:
        raise ValueError("Enter a name between 2 and 100 characters.")
    return value


def _validate_phone(phone_number: str) -> str:
    value = phone_number.strip()
    if not _PHONE_PATTERN.fullmatch(value):
        raise ValueError("Enter a valid phone number.")
    return value


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Hash a new password with Argon2id (salt is accepted for legacy API compatibility)."""
    del salt
    _validate_password(password)
    return _PASSWORD_HASHER.hash(password)


def _verify_legacy_password(password: str, encoded_hash: str) -> bool:
    try:
        salt_hex, expected_hash = encoded_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        actual_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
        ).hex()
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual_hash, expected_hash)


def _verify_password(password: str, stored_hash: Optional[str]) -> bool:
    if not stored_hash:
        return False
    if stored_hash.startswith("$argon2"):
        try:
            return _PASSWORD_HASHER.verify(stored_hash, password)
        except (InvalidHashError, VerificationError, VerifyMismatchError):
            return False
    return _verify_legacy_password(password, stored_hash)


def _public_user(row: sqlite3.Row) -> Dict[str, Any]:
    user = {key: row[key] for key in row.keys() if key not in {"password", "password_hash"}}
    user["is_admin"] = bool(user.get("is_admin", 0))
    user["full_name"] = user.get("name", "")
    user["phone"] = user.get("phone_number")
    return user


def _get_row_by_id(connection: sqlite3.Connection, user_id: int) -> Optional[sqlite3.Row]:
    return connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()


def get_user_by_id(user_id: int, actor: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Return a profile to its owner or an active administrator."""
    with _connect() as connection:
        row = _get_row_by_id(connection, user_id)
        if row is None:
            return None
        if not actor or actor.get("user_id") is None:
            raise PermissionError("An authenticated account is required to view profiles.")
        actor_row = _get_row_by_id(connection, int(actor["user_id"]))
        if actor_row is None or actor_row["account_status"] != ACTIVE_STATUS:
            raise PermissionError("The acting account is unavailable.")
        if actor_row["user_id"] != row["user_id"]:
            _require_admin(connection, actor)
    return _public_user(row) if row else None


def get_user_by_username(
    username: str,
    actor: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return a profile by username/email only to its owner or an administrator."""
    if not actor or actor.get("user_id") is None:
        raise PermissionError("An authenticated account is required to view profiles.")
    identifier = username.strip().casefold()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE lower(username) = ? OR lower(email) = ? LIMIT 1",
            (identifier, identifier),
        ).fetchone()
        if row is not None and int(actor["user_id"]) != row["user_id"]:
            _require_admin(connection, actor)
    return _public_user(row) if row else None


def _record_login_failure(connection: sqlite3.Connection, identifier_hash: str, now: int) -> None:
    row = connection.execute(
        "SELECT failed_attempts, locked_until FROM login_attempts WHERE identifier_hash = ?",
        (identifier_hash,),
    ).fetchone()
    count = 0
    if row:
        if row["locked_until"] > now:
            count = int(row["failed_attempts"])
        elif row["locked_until"] == 0:
            count = int(row["failed_attempts"])
    count += 1
    locked_until = now + AUTH_LOCKOUT_MINUTES * 60 if count >= AUTH_MAX_LOGIN_ATTEMPTS else 0
    connection.execute(
        "INSERT INTO login_attempts(identifier_hash, failed_attempts, locked_until, updated_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(identifier_hash) DO UPDATE SET "
        "failed_attempts=excluded.failed_attempts, locked_until=excluded.locked_until, "
        "updated_at=CURRENT_TIMESTAMP",
        (identifier_hash, count, locked_until),
    )


def authenticate_user(
    identifier: str,
    password: str,
    required_role: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Authenticate by email or legacy username with account-status and lockout checks."""
    if not identifier or not password:
        return None
    normalized = identifier.strip().casefold()
    identifier_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    now = int(time.time())
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        attempt = connection.execute(
            "SELECT locked_until FROM login_attempts WHERE identifier_hash = ?",
            (identifier_hash,),
        ).fetchone()
        if attempt and int(attempt["locked_until"]) > now:
            connection.commit()
            return None

        row = connection.execute(
            "SELECT * FROM users WHERE lower(username) = ? OR lower(email) = ? LIMIT 1",
            (normalized, normalized),
        ).fetchone()
        matches = (
            row is not None
            and row["account_status"] == ACTIVE_STATUS
            and (required_role is None or row["role"] == required_role)
            and _verify_password(password, row["password_hash"] or row["password"])
        )
        if not matches:
            _record_login_failure(connection, identifier_hash, now)
            connection.commit()
            return None

        stored_hash = row["password_hash"] or row["password"]
        if not stored_hash.startswith("$argon2"):
            upgraded_hash = _PASSWORD_HASHER.hash(password)
            connection.execute(
                "UPDATE users SET password = ?, password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                (upgraded_hash, upgraded_hash, row["user_id"]),
            )
        connection.execute("DELETE FROM login_attempts WHERE identifier_hash = ?", (identifier_hash,))
        updated = _get_row_by_id(connection, row["user_id"])
        connection.commit()
        return _public_user(updated)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _insert_user(
    *,
    name: str,
    username: str,
    password: str,
    role: str,
    email: Optional[str] = None,
    phone_number: Optional[str] = None,
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
    designation: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    is_admin: bool = False,
) -> int:
    password_hash = _hash_password(password)
    try:
        with _connect() as connection:
            cursor = connection.execute(
                "INSERT INTO users (name, username, password, password_hash, role, email, phone_number, "
                "employee_id, department, designation, jurisdiction, account_status, is_admin, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                (
                    name, username, password_hash, password_hash, role, email, phone_number,
                    employee_id, department, designation, jurisdiction, ACTIVE_STATUS, int(is_admin),
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as error:
        message = str(error).lower()
        if "email" in message or "username" in message:
            raise ValueError("An account with that email address already exists.") from error
        if "employee_id" in message:
            raise ValueError("That employee identifier is already in use.") from error
        raise


def register_citizen(full_name: str, email: str, phone_number: str, password: str) -> int:
    """Register a citizen using their email as the canonical login identifier."""
    name = _validate_name(full_name)
    normalized_email = _normalise_email(email)
    phone = _validate_phone(phone_number)
    _validate_password(password)
    return _insert_user(
        name=name,
        username=normalized_email,
        password=password,
        role=ROLE_CITIZEN,
        email=normalized_email,
        phone_number=phone,
    )


def register_user(name: str, username: str, password: str, role: str) -> int:
    """Backward-compatible citizen registration; public officer signup is forbidden."""
    if role != ROLE_CITIZEN:
        raise ValueError("Municipal Officer accounts require administrator authorization.")
    clean_name = _validate_name(name)
    clean_username = username.strip()
    if not clean_username:
        raise ValueError("A login email or username is required.")
    _validate_password(password)
    email = clean_username.lower() if _EMAIL_PATTERN.fullmatch(clean_username) else None
    return _insert_user(
        name=clean_name,
        username=clean_username,
        password=password,
        role=ROLE_CITIZEN,
        email=email,
    )


def _require_admin(connection: sqlite3.Connection, actor: Dict[str, Any]) -> sqlite3.Row:
    if not actor or actor.get("user_id") is None:
        raise PermissionError("An authenticated administrator is required.")
    row = _get_row_by_id(connection, int(actor["user_id"]))
    if (
        row is None
        or row["role"] != ROLE_MUNICIPAL_OFFICER
        or not row["is_admin"]
        or row["account_status"] != ACTIVE_STATUS
    ):
        raise PermissionError("Only an active municipal administrator may manage officers.")
    return row


def create_initial_admin(
    *, name: str, email: str, employee_id: str, department: str,
    designation: str, jurisdiction: str, password: str,
) -> int:
    """Create the first administrator exactly once; intended for the local bootstrap CLI."""
    clean_name = _validate_name(name)
    clean_email = _normalise_email(email)
    _validate_password(password)
    if not all(value.strip() for value in (employee_id, department, designation, jurisdiction)):
        raise ValueError("Employee ID, department, designation, and jurisdiction are required.")
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute(
            "SELECT 1 FROM users WHERE role = ? AND is_admin = 1 LIMIT 1",
            (ROLE_MUNICIPAL_OFFICER,),
        ).fetchone():
            raise PermissionError("An initial administrator already exists.")
        password_hash = _hash_password(password)
        cursor = connection.execute(
            "INSERT INTO users (name, username, password, password_hash, role, email, employee_id, department, "
            "designation, jurisdiction, account_status, is_admin, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (
                clean_name, clean_email, password_hash, password_hash, ROLE_MUNICIPAL_OFFICER,
                clean_email, employee_id.strip(), department.strip(), designation.strip(),
                jurisdiction.strip(), ACTIVE_STATUS,
            ),
        )
        user_id = int(cursor.lastrowid)
        connection.commit()
        return user_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def has_active_admin() -> bool:
    """Return whether an active municipal administrator is configured."""
    with _connect() as connection:
        return connection.execute(
            "SELECT 1 FROM users WHERE role = ? AND is_admin = 1 AND account_status = ? LIMIT 1",
            (ROLE_MUNICIPAL_OFFICER, ACTIVE_STATUS),
        ).fetchone() is not None


def create_officer(
    actor: Dict[str, Any], *, name: str, email: str, employee_id: str,
    department: str, designation: str, jurisdiction: str, password: str,
) -> int:
    """Create an active officer only when the persisted actor is an active administrator."""
    clean_name = _validate_name(name)
    clean_email = _normalise_email(email)
    _validate_password(password)
    fields = (employee_id.strip(), department.strip(), designation.strip(), jurisdiction.strip())
    if not all(fields):
        raise ValueError("Employee ID, department, designation, and jurisdiction are required.")
    encoded = _hash_password(password)
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        _require_admin(connection, actor)
        cursor = connection.execute(
            "INSERT INTO users (name, username, password, password_hash, role, email, employee_id, "
            "department, designation, jurisdiction, account_status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (
                clean_name, clean_email, encoded, encoded, ROLE_MUNICIPAL_OFFICER,
                clean_email, fields[0], fields[1], fields[2], fields[3], ACTIVE_STATUS,
            ),
        )
        user_id = int(cursor.lastrowid)
        connection.commit()
        return user_id
    except sqlite3.IntegrityError as error:
        connection.rollback()
        if "employee_id" in str(error).lower():
            raise ValueError("That employee identifier is already in use.") from error
        raise ValueError("An account with that email address already exists.") from error
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def list_officers(actor: Dict[str, Any]) -> list[Dict[str, Any]]:
    """List officer profiles for administrator-only management views."""
    with _connect() as connection:
        _require_admin(connection, actor)
        rows = connection.execute(
            "SELECT * FROM users WHERE role = ? ORDER BY name COLLATE NOCASE",
            (ROLE_MUNICIPAL_OFFICER,),
        ).fetchall()
    return [_public_user(row) for row in rows]


def update_officer(actor: Dict[str, Any], user_id: int, *, account_status: str, is_admin: bool = False) -> None:
    """Activate/deactivate an officer and manage admin privileges safely."""
    status = account_status.strip().lower()
    if status not in {ACTIVE_STATUS, INACTIVE_STATUS}:
        raise ValueError("Account status must be active or inactive.")
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        _require_admin(connection, actor)
        target = _get_row_by_id(connection, user_id)
        if target is None or target["role"] != ROLE_MUNICIPAL_OFFICER:
            raise ValueError("Municipal officer account not found.")
        if target["user_id"] == actor["user_id"] and (status != ACTIVE_STATUS or not is_admin):
            raise ValueError("You cannot deactivate or remove admin access from your own account.")
        if target["is_admin"] and (status != ACTIVE_STATUS or not is_admin):
            remaining = connection.execute(
                "SELECT COUNT(*) FROM users WHERE role = ? AND is_admin = 1 AND account_status = ? AND user_id <> ?",
                (ROLE_MUNICIPAL_OFFICER, ACTIVE_STATUS, user_id),
            ).fetchone()[0]
            if remaining == 0:
                raise ValueError("At least one active administrator must remain.")
        connection.execute(
            "UPDATE users SET account_status = ?, is_admin = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
            (status, int(is_admin), user_id),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def update_user_profile(
    actor: Dict[str, Any], user_id: int, *, full_name: str, email: str, phone_number: str
) -> None:
    """Update only the current user's basic profile fields."""
    if not actor or int(actor.get("user_id", -1)) != int(user_id):
        raise PermissionError("You may update only your own profile.")
    name = _validate_name(full_name)
    normalized_email = _normalise_email(email)
    phone = _validate_phone(phone_number)
    try:
        with _connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET name = ?, email = ?, username = CASE WHEN username = email THEN ? ELSE username END, "
                "phone_number = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND account_status = ?",
                (name, normalized_email, normalized_email, phone, user_id, ACTIVE_STATUS),
            )
            if cursor.rowcount == 0:
                raise ValueError("Profile could not be updated.")
    except sqlite3.IntegrityError as error:
        raise ValueError("That email address is already associated with another account.") from error


def change_password(actor: Dict[str, Any], current_password: str, new_password: str) -> None:
    """Verify the current password and replace it with a new Argon2id hash."""
    if not actor or actor.get("user_id") is None:
        raise PermissionError("Sign in before changing your password.")
    _validate_password(new_password)
    with _connect() as connection:
        row = _get_row_by_id(connection, int(actor["user_id"]))
        if row is None or row["account_status"] != ACTIVE_STATUS:
            raise PermissionError("The account is not active.")
        if not _verify_password(current_password, row["password_hash"] or row["password"]):
            raise ValueError("Current password is incorrect.")
        encoded = _PASSWORD_HASHER.hash(new_password)
        connection.execute(
            "UPDATE users SET password = ?, password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
            (encoded, encoded, row["user_id"]),
        )
        connection.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (row["user_id"],))


def password_reset_delivery_configured() -> bool:
    """Return whether the required SMTP settings are available."""
    return bool(SMTP_HOST and SMTP_FROM_EMAIL)


def request_password_reset(email: str) -> bool:
    """Issue a one-use, expiring reset token and deliver it by configured SMTP."""
    if not password_reset_delivery_configured():
        return False
    try:
        normalized_email = _normalise_email(email)
    except ValueError:
        return False
    with _connect() as connection:
        row = connection.execute(
            "SELECT user_id, email FROM users WHERE lower(email) = ? AND account_status = ?",
            (normalized_email, ACTIVE_STATUS),
        ).fetchone()
        if row is None:
            return False
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expires_at = int(time.time()) + PASSWORD_RESET_TTL_MINUTES * 60
        connection.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (row["user_id"],))
        connection.execute(
            "INSERT INTO password_reset_tokens(user_id, token_hash, expires_at) VALUES (?, ?, ?)",
            (row["user_id"], token_hash, expires_at),
        )

    reset_link = f"{PASSWORD_RESET_PUBLIC_URL.rstrip('/')}?reset={quote(token)}"
    message = EmailMessage()
    message["Subject"] = "CiviScan password reset"
    message["From"] = SMTP_FROM_EMAIL
    message["To"] = normalized_email
    message.set_content(
        f"Use this one-time link to reset your CiviScan password:\n\n{reset_link}\n\n"
        f"The link expires in {PASSWORD_RESET_TTL_MINUTES} minutes. If you did not request it, ignore this email."
    )
    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(), timeout=15) as client:
                if SMTP_USERNAME:
                    client.login(SMTP_USERNAME, SMTP_PASSWORD)
                client.send_message(message)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as client:
                client.starttls(context=ssl.create_default_context())
                if SMTP_USERNAME:
                    client.login(SMTP_USERNAME, SMTP_PASSWORD)
                client.send_message(message)
    except (OSError, smtplib.SMTPException):
        with _connect() as connection:
            connection.execute("DELETE FROM password_reset_tokens WHERE token_hash = ?", (token_hash,))
        return False
    return True


def reset_password(token: str, new_password: str) -> bool:
    """Consume a one-time reset token and update the user's password hash."""
    _validate_password(new_password)
    if not token:
        return False
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = int(time.time())
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT user_id FROM password_reset_tokens WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?",
            (token_hash, now),
        ).fetchone()
        if row is None:
            connection.commit()
            return False
        encoded = _PASSWORD_HASHER.hash(new_password)
        connection.execute(
            "UPDATE users SET password = ?, password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND account_status = ?",
            (encoded, encoded, row["user_id"], ACTIVE_STATUS),
        )
        connection.execute("UPDATE password_reset_tokens SET used_at = ? WHERE token_hash = ?", (now, token_hash))
        connection.execute("DELETE FROM login_attempts")
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def establish_session(session_state, user: Dict[str, Any]) -> None:
    """Store only public identity fields in the current Streamlit session."""
    session_state["authenticated_user_id"] = int(user["user_id"])
    session_state["current_user"] = user
    session_state["logged_in"] = True
    session_state["user_id"] = int(user["user_id"])
    session_state["username"] = user.get("username")
    session_state["role"] = user.get("role")
    session_state["name"] = user.get("name")


def get_current_user(session_state) -> Optional[Dict[str, Any]]:
    """Reload the account on each rerun, invalidating sessions for disabled users."""
    user_id = session_state.get("authenticated_user_id") or session_state.get("user_id")
    if not session_state.get("logged_in") or user_id is None:
        return None
    with _connect() as connection:
        row = _get_row_by_id(connection, int(user_id))
    user = _public_user(row) if row else None
    if user is None or user.get("account_status") != ACTIVE_STATUS:
        clear_session(session_state)
        return None
    establish_session(session_state, user)
    return user


def clear_session(session_state) -> None:
    """Remove all authentication identity values from the Streamlit session."""
    for key in (
        "authenticated_user_id", "current_user", "logged_in", "user_id", "username", "role", "name",
    ):
        session_state.pop(key, None)


def require_authenticated_role(user: Optional[Dict[str, Any]], allowed_roles: Iterable[str]) -> Dict[str, Any]:
    """Check the current persisted account role/status for a protected operation."""
    from civiscan.security import require_role

    if not user or user.get("user_id") is None:
        raise PermissionError("An authenticated account is required.")
    with _connect() as connection:
        row = _get_row_by_id(connection, int(user["user_id"]))
    if row is None:
        raise PermissionError("The account is unavailable.")
    return require_role(_public_user(row), allowed_roles)
