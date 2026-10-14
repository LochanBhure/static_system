"""Tests for alert-window shift report assembly and persistence."""

from datetime import datetime, timedelta
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock
import unittest

from db.database import Database
from shift_report import ShiftReportGenerator


class ShiftReportTests(unittest.TestCase):
    def test_existing_alert_database_gets_camera_id_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            with closing(sqlite3.connect(path)) as connection:
                with connection:
                    connection.execute(
                        """
                        CREATE TABLE alerts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            created_at TEXT NOT NULL,
                            frame_number INTEGER,
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

            Database(path)

            with closing(sqlite3.connect(path)) as connection:
                columns = {
                    row[1] for row in connection.execute(
                        "PRAGMA table_info(alerts)"
                    )
                }
            self.assertIn("camera_id", columns)

    def test_report_collects_alerts_saves_json_and_dispatches_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "cctv.sqlite3")
            database.insert_alert(
                {
                    "severity": "critical",
                    "title": "Restricted-zone entry",
                    "message": "Person entered secure area.",
                    "objects": ["cam1:p1"],
                },
                {
                    "level": "HIGH",
                    "frame_number": 18,
                    "coordinates": {"cam1:p1": {"x": 400, "y": 200}},
                    "snapshot_path": "/tmp/captured.jpg",
                },
            )
            database.insert_alert(
                {
                    "severity": "warning",
                    "title": "Unattended item",
                    "message": "Package left alone.",
                    "objects": ["cam1:item1"],
                },
                {
                    "level": "MEDIUM",
                    "frame_number": 19,
                    "coordinates": {},
                    "snapshot_path": None,
                },
            )
            hook = Mock()
            reporter = ShiftReportGenerator(
                database=database,
                report_hooks=[hook],
            )
            reporter._summarize = Mock(return_value={
                "summary": "Two events were recorded.",
                "anomalies": ["Restricted-zone entry"],
            })
            period_end = datetime.now().astimezone() + timedelta(seconds=5)
            period_start = period_end - timedelta(minutes=1)

            report = reporter.generate_report(period_start, period_end)

            self.assertEqual(report["total_events"], 2)
            self.assertEqual(len(report["high_alerts"]), 1)
            self.assertEqual(len(report["medium_alerts"]), 1)
            self.assertEqual(report["summary"], "Two events were recorded.")
            hook.assert_called_once_with(report)
            with closing(sqlite3.connect(database.path)) as connection:
                stored = connection.execute(
                    "SELECT report_json FROM shift_reports"
                ).fetchone()
            self.assertEqual(stored[0], json.dumps(report, sort_keys=True))

    def test_report_requires_aware_period(self):
        reporter = ShiftReportGenerator(database=Mock())
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            reporter.generate_report(
                datetime(2026, 1, 1),
                datetime(2026, 1, 2),
            )


if __name__ == "__main__":
    unittest.main()
