"""Small SQLite repository shared by alert delivery and shift reporting."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import config


class Database:
    """Persist CCTV alert records and generated shift reports."""

    def __init__(self, path=None):
        self.path = Path(path or config.ALERT_DATABASE_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self):
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        frame_number INTEGER,
                        camera_id TEXT,
                        level TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        objects_json TEXT NOT NULL,
                        coordinates_json TEXT NOT NULL,
                        snapshot_path TEXT
                    )
                    """
                )
                alert_columns = {
                    row[1] for row in connection.execute(
                        "PRAGMA table_info(alerts)"
                    )
                }
                if "camera_id" not in alert_columns:
                    connection.execute(
                        "ALTER TABLE alerts ADD COLUMN camera_id TEXT"
                    )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS shift_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        period_start TEXT NOT NULL,
                        period_end TEXT NOT NULL,
                        total_events INTEGER NOT NULL,
                        report_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )

    def insert_alert(self, alert, context):
        """Store the alert record and related metadata."""
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO alerts (
                        created_at, frame_number, camera_id, level, severity,
                        title, message, objects_json, coordinates_json,
                        snapshot_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().astimezone().isoformat(),
                        context.get("frame_number"),
                        context.get("camera_id"),
                        context["level"],
                        alert.get("severity", alert.get("alert_level", "")),
                        alert.get("title", alert.get("alert_type", "")),
                        alert.get("message", alert.get("reason", "")),
                        json.dumps(
                            alert.get(
                                "objects",
                                alert.get("affected_object_ids", []),
                            )
                        ),
                        json.dumps(context.get("coordinates", {}), sort_keys=True),
                        context.get("snapshot_path"),
                    ),
                )

    def get_alerts(self, period_start, period_end):
        """Read alert rows whose ISO timestamps fall inside the requested window."""
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, created_at, frame_number, camera_id, level, severity,
                       title, message, objects_json, coordinates_json,
                       snapshot_path
                FROM alerts
                WHERE julianday(created_at) >= julianday(?)
                  AND julianday(created_at) < julianday(?)
                ORDER BY julianday(created_at), id
                """,
                (period_start.isoformat(), period_end.isoformat()),
            ).fetchall()
        return [
            {
                **dict(row),
                "objects": json.loads(row["objects_json"]),
                "coordinates": json.loads(row["coordinates_json"]),
            }
            for row in rows
        ]

    def save_report(self, report):
        """Persist a shift report as JSON and return its row ID."""
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO shift_reports (
                        period_start, period_end, total_events, report_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        report["period_start"],
                        report["period_end"],
                        report["total_events"],
                        json.dumps(report, sort_keys=True),
                        datetime.now().astimezone().isoformat(),
                    ),
                )
                return int(cursor.lastrowid)
