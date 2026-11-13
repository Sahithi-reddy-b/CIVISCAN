"""SQLite persistence for CiviScan civic issue reports."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATABASE_PATH
from civiscan.security import AuthorizationError

DATABASE_PATH = Path(DATABASE_PATH)


ISSUES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS issues (
    issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    category TEXT NOT NULL,
    confidence REAL,
    severity TEXT,
    estimated_repair_cost REAL,
    latitude REAL,
    longitude REAL,
    reported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT,
    status TEXT NOT NULL DEFAULT 'NEW',
    priority_score REAL,
    priority_label TEXT,
    description TEXT
)
"""

ISSUE_MIGRATIONS = {
    "source": "TEXT NOT NULL DEFAULT 'citizen'",
    "severity_reason": "TEXT",
    "priority_reason": "TEXT",
    "annotated_image_path": "TEXT",
    "model_name": "TEXT",
    "model_version": "TEXT",
    "inference_seconds": "REAL",
    "updated_at": "TEXT",
    "assigned_team": "TEXT",
    "repair_notes": "TEXT",
    "resolved_at": "TEXT",
    "repair_actual_cost": "REAL",
    "repair_before_image": "TEXT",
    "repair_after_image": "TEXT",
}

MEDIA_TABLES_SQL = (
    """
    CREATE TABLE IF NOT EXISTS processing_jobs (
        job_id TEXT PRIMARY KEY,
        source_filename TEXT NOT NULL,
        original_path TEXT NOT NULL,
        content_sha256 TEXT NOT NULL,
        media_type TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
        processed_path TEXT,
        error_message TEXT,
        detection_count INTEGER NOT NULL DEFAULT 0,
        frames_processed INTEGER NOT NULL DEFAULT 0,
        started_at TEXT,
        completed_at TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS detections (
        detection_id TEXT PRIMARY KEY,
        job_id TEXT REFERENCES processing_jobs(job_id) ON DELETE SET NULL,
        issue_id INTEGER REFERENCES issues(issue_id) ON DELETE SET NULL,
        image_filename TEXT NOT NULL,
        issue_category TEXT NOT NULL,
        class_id INTEGER NOT NULL,
        confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
        bbox_json TEXT NOT NULL,
        captured_at TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        inference_seconds REAL NOT NULL,
        annotated_image_path TEXT,
        frame_timestamp_seconds REAL,
        severity TEXT,
        severity_reason TEXT,
        estimated_repair_cost REAL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS media_assets (
        asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id INTEGER REFERENCES issues(issue_id) ON DELETE SET NULL,
        job_id TEXT REFERENCES processing_jobs(job_id) ON DELETE SET NULL,
        asset_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS issue_status_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id INTEGER NOT NULL REFERENCES issues(issue_id) ON DELETE CASCADE,
        actor_user_id INTEGER,
        old_status TEXT,
        new_status TEXT NOT NULL,
        notes TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS repair_assignments (
        assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id INTEGER NOT NULL REFERENCES issues(issue_id) ON DELETE CASCADE,
        assigned_team TEXT NOT NULL,
        assigned_by INTEGER,
        notes TEXT,
        assigned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS repair_verification (
        verification_id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id INTEGER NOT NULL REFERENCES issues(issue_id) ON DELETE CASCADE,
        before_image_path TEXT,
        after_image_path TEXT,
        decision TEXT NOT NULL DEFAULT 'pending',
        notes TEXT,
        reviewed_by INTEGER,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        reviewed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notifications (
        notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_id INTEGER REFERENCES issues(issue_id) ON DELETE SET NULL,
        recipient_user_id INTEGER,
        channel TEXT NOT NULL,
        message TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        dedupe_key TEXT UNIQUE,
        error_message TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        delivered_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS maintenance_cost_rates (
        category TEXT PRIMARY KEY,
        material REAL NOT NULL CHECK(material >= 0),
        labour REAL NOT NULL CHECK(labour >= 0),
        equipment REAL NOT NULL CHECK(equipment >= 0),
        transportation REAL NOT NULL CHECK(transportation >= 0),
        updated_by INTEGER,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_evaluations (
        evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        dataset_version TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        evaluated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
)


class _ClosingConnection(sqlite3.Connection):
    """Close database handles when a transaction context exits."""

    def __exit__(self, exception_type, exception, traceback):
        try:
            return super().__exit__(exception_type, exception, traceback)
        finally:
            self.close()


def _connect() -> sqlite3.Connection:
    """Open the application database with dictionary-like row access."""
    connection = sqlite3.connect(DATABASE_PATH, factory=_ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _require_persisted_role(connection, actor, allowed_roles):
    """Authorize using the database's current role/status, not caller-supplied claims."""
    if not actor or actor.get("user_id") is None:
        raise AuthorizationError("An authenticated account is required.")
    row = connection.execute(
        "SELECT role, account_status FROM users WHERE user_id = ?",
        (int(actor["user_id"]),),
    ).fetchone()
    if row is None or row["account_status"] != "active":
        raise AuthorizationError("The account is unavailable.")
    if row["role"] not in set(allowed_roles):
        raise AuthorizationError("You are not authorized to perform this action.")
    return row


def initialize_database() -> None:
    """Create or additively migrate the civic issue and media-processing schema."""
    with _connect() as connection:
        connection.execute(ISSUES_TABLE_SQL)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(issues)")}
        for column, definition in ISSUE_MIGRATIONS.items():
            if column not in columns:
                connection.execute(f"ALTER TABLE issues ADD COLUMN {column} {definition}")
        for statement in MEDIA_TABLES_SQL:
            connection.execute(statement)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_issues_category_date ON issues(category, reported_at)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status, created_at)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_jobs_sha ON processing_jobs(content_sha256)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_detections_issue ON detections(issue_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_detections_job ON detections(job_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_status_history_issue ON issue_status_history(issue_id, created_at)")


def create_processing_job(source_filename: str, original_path: str, content_sha256: str, media_type: str) -> str:
    """Persist a new incoming-media processing job."""
    if not source_filename or not content_sha256 or media_type not in {"image", "video"}:
        raise ValueError("A filename, SHA-256 digest, and supported media type are required.")
    job_id = uuid.uuid4().hex
    with _connect() as connection:
        connection.execute(
            """INSERT INTO processing_jobs
            (job_id, source_filename, original_path, content_sha256, media_type, status)
            VALUES (?, ?, ?, ?, ?, 'pending')""",
            (job_id, source_filename, original_path, content_sha256, media_type),
        )
    return job_id


def mark_processing_job_started(job_id: str) -> bool:
    """Move a queued processing job from pending to processing."""
    with _connect() as connection:
        cursor = connection.execute(
            """UPDATE processing_jobs SET status = 'processing', started_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status = 'pending'""",
            (job_id,),
        )
    return cursor.rowcount > 0


def update_processing_job(
    job_id: str,
    status: str,
    *,
    processed_path: Optional[str] = None,
    error_message: Optional[str] = None,
    detection_count: int = 0,
    frames_processed: int = 0,
) -> bool:
    """Record a terminal processing state and its output summary."""
    if status not in {"completed", "failed"}:
        raise ValueError("A processing job can only be completed or failed here.")
    with _connect() as connection:
        cursor = connection.execute(
            """UPDATE processing_jobs SET status = ?, processed_path = ?, error_message = ?,
            detection_count = ?, frames_processed = ?, completed_at = CURRENT_TIMESTAMP
            WHERE job_id = ?""",
            (status, processed_path, error_message, int(detection_count), int(frames_processed), job_id),
        )
    return cursor.rowcount > 0


def completed_job_for_hash(content_sha256: str) -> Optional[Dict[str, Any]]:
    """Return a previously completed job with identical media bytes, if any."""
    with _connect() as connection:
        row = connection.execute(
            """SELECT * FROM processing_jobs WHERE content_sha256 = ? AND status = 'completed'
            ORDER BY completed_at DESC LIMIT 1""",
            (content_sha256,),
        ).fetchone()
    return dict(row) if row is not None else None


def get_processing_jobs(limit: int = 100) -> List[Dict[str, Any]]:
    """Return recent incoming-media job history."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM processing_jobs ORDER BY created_at DESC LIMIT ?",
            (max(1, min(int(limit), 1000)),),
        ).fetchall()
    return [dict(row) for row in rows]


def get_media_statistics() -> Dict[str, int]:
    """Return persisted processing totals, grouped by media type and state."""
    with _connect() as connection:
        row = connection.execute(
            """SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN media_type = 'image' AND status = 'completed' THEN 1 ELSE 0 END) AS images_processed,
                SUM(CASE WHEN media_type = 'video' AND status = 'completed' THEN 1 ELSE 0 END) AS videos_processed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_jobs,
                SUM(CASE WHEN status IN ('pending', 'processing') THEN 1 ELSE 0 END) AS active_jobs,
                SUM(detection_count) AS detections
            FROM processing_jobs"""
        ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()}


def get_detection_count() -> int:
    """Return the number of persisted model detections."""
    with _connect() as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM detections").fetchone()
    return int(row["count"])


def get_maintenance_cost_rates() -> Dict[str, Dict[str, float]]:
    """Return configured category-level indicative maintenance rates."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT category, material, labour, equipment, transportation FROM maintenance_cost_rates"
        ).fetchall()
    return {
        row["category"]: {
            "material": float(row["material"]),
            "labour": float(row["labour"]),
            "equipment": float(row["equipment"]),
            "transportation": float(row["transportation"]),
        }
        for row in rows
    }


def set_maintenance_cost_rates(category: str, rates: Dict[str, float], actor=None) -> None:
    """Set category-level cost assumptions for an authorized municipal officer."""
    required = ("material", "labour", "equipment", "transportation")
    if not category or any(key not in rates for key in required):
        raise ValueError("Provide a category and all four cost components.")
    values = [float(rates[key]) for key in required]
    if any(value < 0 for value in values):
        raise ValueError("Maintenance cost rates cannot be negative.")
    if not any(value > 0 for value in values):
        raise ValueError("At least one indicative cost component must be greater than zero.")
    with _connect() as connection:
        _require_persisted_role(connection, actor, {"Municipal Officer"})
        connection.execute(
            """INSERT INTO maintenance_cost_rates
            (category, material, labour, equipment, transportation, updated_by, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(category) DO UPDATE SET
            material = excluded.material, labour = excluded.labour,
            equipment = excluded.equipment, transportation = excluded.transportation,
            updated_by = excluded.updated_by, updated_at = CURRENT_TIMESTAMP""",
            (category, *values, int(actor["user_id"])),
        )


def get_processed_content_paths() -> set:
    """Return original paths already represented by a completed job."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT original_path FROM processing_jobs WHERE status = 'completed'"
        ).fetchall()
    return {str(row["original_path"]) for row in rows}


def record_worker_detection(
    job_id: str,
    detection: Dict[str, Any],
    annotated_image_path: str,
    *,
    frame_timestamp_seconds: Optional[float] = None,
    estimated_repair_cost: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> int:
    """Persist a camera detection and its actionable civic issue in one transaction."""
    category = str(detection["issue_category"])
    bbox = [float(value) for value in detection["bbox"]]
    captured_at = str(detection.get("timestamp") or "")
    with _connect() as connection:
        cursor = connection.execute(
            """INSERT INTO issues
            (user_id, category, confidence, severity, estimated_repair_cost, latitude, longitude,
             reported_at, image_path, status, source, severity_reason, annotated_image_path, model_name,
             model_version, inference_seconds, updated_at)
            VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, 'NEW', 'incoming_media', ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (
                category,
                float(detection["confidence"]),
                detection.get("severity"),
                estimated_repair_cost,
                latitude,
                longitude,
                captured_at,
                str(detection.get("source_path") or ""),
                detection.get("severity_reason"),
                annotated_image_path,
                str(detection["model_name"]),
                str(detection["model_version"]),
                float(detection["inference_seconds"]),
            ),
        )
        issue_id = int(cursor.lastrowid)
        connection.execute(
            """INSERT INTO detections
            (detection_id, job_id, issue_id, image_filename, issue_category, class_id, confidence,
             bbox_json, captured_at, model_name, model_version, inference_seconds,
             annotated_image_path, frame_timestamp_seconds, severity, severity_reason,
             estimated_repair_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(detection["detection_id"]),
                job_id,
                issue_id,
                str(detection["image_filename"]),
                category,
                int(detection["class_id"]),
                float(detection["confidence"]),
                json.dumps(bbox),
                captured_at,
                str(detection["model_name"]),
                str(detection["model_version"]),
                float(detection["inference_seconds"]),
                annotated_image_path,
                frame_timestamp_seconds,
                detection.get("severity"),
                detection.get("severity_reason"),
                estimated_repair_cost,
            ),
        )
        connection.execute(
            """INSERT INTO media_assets(issue_id, job_id, asset_type, file_path)
            SELECT ?, ?, 'annotated', ? WHERE NOT EXISTS
            (SELECT 1 FROM media_assets WHERE asset_type = 'annotated' AND file_path = ?)""",
            (issue_id, job_id, annotated_image_path, annotated_image_path),
        )
        connection.execute(
            """INSERT OR IGNORE INTO media_assets(issue_id, job_id, asset_type, file_path)
            SELECT ?, ?, 'source', ? WHERE NOT EXISTS
            (SELECT 1 FROM media_assets WHERE asset_type = 'source' AND file_path = ?)""",
            (
                issue_id,
                job_id,
                str(detection.get("source_path") or ""),
                str(detection.get("source_path") or ""),
            ),
        )
    return issue_id


def record_inference_detections(
    detections: List[Dict[str, Any]],
    annotated_image_path: str,
    actor,
    source_image_path: Optional[str] = None,
) -> List[str]:
    """Persist citizen/officer image predictions without creating a civic issue."""
    detection_ids = []
    with _connect() as connection:
        _require_persisted_role(connection, actor, {"Citizen", "Municipal Officer"})
        for detection in detections:
            detection_id = str(detection["detection_id"])
            connection.execute(
                """INSERT OR IGNORE INTO detections
                (detection_id, image_filename, issue_category, class_id, confidence, bbox_json,
                 captured_at, model_name, model_version, inference_seconds, annotated_image_path,
                 severity, severity_reason, estimated_repair_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    detection_id,
                    str(detection["image_filename"]),
                    str(detection["issue_category"]),
                    int(detection["class_id"]),
                    float(detection["confidence"]),
                    json.dumps([float(value) for value in detection["bbox"]]),
                    str(detection["timestamp"]),
                    str(detection["model_name"]),
                    str(detection["model_version"]),
                    float(detection["inference_seconds"]),
                    annotated_image_path,
                    detection.get("severity"),
                    detection.get("severity_reason"),
                    detection.get("estimated_repair_cost"),
                ),
            )
            detection_ids.append(detection_id)
        if detections:
            connection.execute(
                """INSERT INTO media_assets(asset_type, file_path)
                SELECT 'annotated', ? WHERE NOT EXISTS
                (SELECT 1 FROM media_assets WHERE asset_type = 'annotated' AND file_path = ?)""",
                (annotated_image_path, annotated_image_path),
            )
            if source_image_path:
                connection.execute(
                    """INSERT INTO media_assets(asset_type, file_path)
                    SELECT 'source', ? WHERE NOT EXISTS
                    (SELECT 1 FROM media_assets WHERE asset_type = 'source' AND file_path = ?)""",
                    (source_image_path, source_image_path),
                )
    return detection_ids


def link_detections_to_issue(detection_ids: List[str], issue_id: int, actor) -> int:
    """Associate saved predictions with an issue owned by the citizen or managed by an officer."""
    if not detection_ids:
        return 0
    placeholders = ", ".join("?" for _ in detection_ids)
    with _connect() as connection:
        user = _require_persisted_role(
            connection, actor, {"Citizen", "Municipal Officer"}
        )
        issue = connection.execute(
            "SELECT user_id FROM issues WHERE issue_id = ?", (issue_id,)
        ).fetchone()
        if issue is None:
            raise ValueError("Issue does not exist.")
        if user["role"] == "Citizen" and str(issue["user_id"]) != str(actor["user_id"]):
            raise AuthorizationError("Citizens may link detections only to their own reports.")
        cursor = connection.execute(
            f"UPDATE detections SET issue_id = ? WHERE detection_id IN ({placeholders})",
            [issue_id, *detection_ids],
        )
    return cursor.rowcount


def add_issue(
    user_id: Optional[str],
    category: str,
    confidence: Optional[float] = None,
    severity: Optional[str] = None,
    estimated_repair_cost: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    reported_at: Optional[str] = None,
    image_path: Optional[str] = None,
    status: str = "NEW",
    priority_score: Optional[float] = None,
    priority_label: Optional[str] = None,
    description: Optional[str] = None,
    severity_reason: Optional[str] = None,
    annotated_image_path: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    inference_seconds: Optional[float] = None,
    actor=None,
) -> int:
    """Insert one issue and return its generated issue ID."""
    if actor is None:
        raise AuthorizationError("An authenticated citizen is required to submit a report.")
    initialize_database()
    columns = [
        "user_id",
        "category",
        "confidence",
        "severity",
        "estimated_repair_cost",
        "latitude",
        "longitude",
        "image_path",
        "status",
        "priority_score",
        "priority_label",
        "description",
        "severity_reason",
        "annotated_image_path",
        "model_name",
        "model_version",
        "inference_seconds",
    ]
    values = [
        user_id,
        category,
        confidence,
        severity,
        estimated_repair_cost,
        latitude,
        longitude,
        image_path,
        status,
        priority_score,
        priority_label,
        description,
        severity_reason,
        annotated_image_path,
        model_name,
        model_version,
        inference_seconds,
    ]

    if reported_at is not None:
        columns.append("reported_at")
        values.append(reported_at)

    placeholders = ", ".join("?" for _ in values)
    column_sql = ", ".join(columns)
    query = f"INSERT INTO issues ({column_sql}) VALUES ({placeholders})"

    with _connect() as connection:
        _require_persisted_role(connection, actor, {"Citizen"})
        if str(user_id) != str(actor["user_id"]):
            raise AuthorizationError("Citizens may submit reports only for their own account.")
        cursor = connection.execute(query, values)
        return int(cursor.lastrowid)


def get_all_issues(actor=None) -> List[Dict[str, Any]]:
    """Return all issues, newest reports first."""
    query = "SELECT * FROM issues ORDER BY reported_at DESC, issue_id DESC"
    with _connect() as connection:
        _require_persisted_role(connection, actor, {"Municipal Officer"})
        rows = connection.execute(query).fetchall()
    return [dict(row) for row in rows]


def get_issue_by_id(issue_id: int, actor=None) -> Optional[Dict[str, Any]]:
    """Return an issue only to its citizen owner or an active municipal officer."""
    query = "SELECT * FROM issues WHERE issue_id = ?"
    with _connect() as connection:
        row = connection.execute(query, (issue_id,)).fetchone()
        if row is not None:
            user = _require_persisted_role(
                connection,
                actor,
                {"Citizen", "Municipal Officer"},
            )
            if user["role"] == "Citizen" and str(row["user_id"]) != str(actor["user_id"]):
                raise AuthorizationError("Citizens may view only their own reports.")
    return dict(row) if row is not None else None


def update_issue_status(issue_id: int, status: str, actor=None) -> bool:
    """Update an issue status after verifying the acting municipal officer."""
    if actor is None:
        raise AuthorizationError("An authenticated municipal officer is required.")
    if status not in {
        "NEW",
        "VERIFIED",
        "ASSIGNED",
        "IN PROGRESS",
        "AWAITING VERIFICATION",
        "RESOLVED",
        "REJECTED",
        "REOPENED",
    }:
        raise ValueError("Invalid issue status.")
    query = "UPDATE issues SET status = ? WHERE issue_id = ?"
    with _connect() as connection:
        _require_persisted_role(connection, actor, {"Municipal Officer"})
        current = connection.execute("SELECT status FROM issues WHERE issue_id = ?", (issue_id,)).fetchone()
        if current is None:
            return False
        cursor = connection.execute(query, (status, issue_id))
        if cursor.rowcount:
            connection.execute(
                """INSERT INTO issue_status_history
                (issue_id, actor_user_id, old_status, new_status)
                VALUES (?, ?, ?, ?)""",
                (issue_id, int(actor["user_id"]), current["status"], status),
            )
    return cursor.rowcount > 0


def update_issue_priority(
    issue_id: int,
    priority_score: float,
    priority_label: str,
    actor=None,
    reason: Optional[str] = None,
) -> bool:
    """Update the calculated priority fields for an existing issue."""
    if actor is None:
        raise AuthorizationError("An authenticated municipal officer is required.")
    with _connect() as connection:
        _require_persisted_role(connection, actor, {"Municipal Officer"})
        cursor = connection.execute(
            """UPDATE issues SET priority_score = ?, priority_label = ?, priority_reason = ?
            WHERE issue_id = ?""",
            (priority_score, priority_label, reason, issue_id),
        )
    return cursor.rowcount > 0


def get_issues_by_user(user_id: str, actor=None) -> List[Dict[str, Any]]:
    """Return all issues submitted by one user, newest reports first."""
    if actor is None:
        raise AuthorizationError("An authenticated citizen is required.")
    query = "SELECT * FROM issues WHERE user_id = ? ORDER BY reported_at DESC, issue_id DESC"
    with _connect() as connection:
        _require_persisted_role(connection, actor, {"Citizen"})
        if str(user_id) != str(actor["user_id"]):
            raise AuthorizationError("Citizens may view only their own reports.")
        rows = connection.execute(query, (user_id,)).fetchall()
    return [dict(row) for row in rows]
