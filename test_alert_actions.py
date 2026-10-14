"""Tests for severity-based alert persistence and action routing."""

import json
import sqlite3
import tempfile
from contextlib import closing
import unittest
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import numpy as np

from reasoning_pipeline import OutputAlertLayer


class OutputAlertLayerTests(unittest.TestCase):
    def test_severity_routes_actions_and_persists_frame_and_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            push = Mock()
            email = Mock()
            buzzer = Mock()
            notification = Mock()
            layer = OutputAlertLayer(
                hooks={
                    "push": push,
                    "email": email,
                    "buzzer": buzzer,
                    "notification": notification,
                },
                database_path=root / "alerts.sqlite3",
                frame_directory=root / "frames",
            )
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            coordinates = {"Person1": {"x": 12, "y": 10}}
            alerts = [
                {
                    "severity": "critical",
                    "title": "Urgent event",
                    "message": "Immediate response required.",
                    "objects": ["Person1"],
                },
                {
                    "severity": "warning",
                    "title": "Notice",
                    "message": "Review this event.",
                    "objects": ["Person1"],
                },
                {
                    "severity": "info",
                    "title": "Observation",
                    "message": "Record this event.",
                    "objects": [],
                },
            ]

            layer.emit_alerts(
                alerts,
                frame=frame,
                coordinates=coordinates,
                frame_number=42,
            )

            for hook in (push, email, buzzer):
                hook.assert_called_once()
                self.assertEqual(hook.call_args.args[1]["level"], "HIGH")
                self.assertTrue(
                    Path(hook.call_args.args[1]["snapshot_path"]).is_file()
                )
            notification.assert_called_once()
            self.assertEqual(
                notification.call_args.args[1]["level"], "MEDIUM"
            )

            with closing(sqlite3.connect(root / "alerts.sqlite3")) as connection:
                rows = connection.execute(
                    """
                    SELECT level, frame_number, coordinates_json, snapshot_path
                    FROM alerts ORDER BY id
                    """
                ).fetchall()

            self.assertEqual([row[0] for row in rows], ["HIGH", "MEDIUM", "LOW"])
            self.assertEqual(rows[0][1], 42)
            self.assertEqual(
                json.loads(rows[0][2]), {"Person1": {"x": 12, "y": 10}}
            )
            self.assertTrue(Path(rows[0][3]).is_file())
            self.assertEqual(rows[1][3], None)
            self.assertEqual(rows[2][3], None)

    def test_classifier_rejects_unknown_severity(self):
        with self.assertRaisesRegex(ValueError, "Unsupported alert severity"):
            OutputAlertLayer.classify_severity("urgent")

    def test_high_alert_requires_a_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            layer = OutputAlertLayer(
                database_path=Path(directory) / "alerts.sqlite3",
                frame_directory=Path(directory) / "frames",
            )
            with self.assertRaisesRegex(ValueError, "frame is required"):
                layer.emit_alerts(
                    [{
                        "severity": "critical",
                        "title": "Urgent event",
                        "message": "Immediate response required.",
                        "objects": [],
                    }]
                )

    def test_failed_high_channel_does_not_skip_remaining_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            push = Mock(side_effect=URLError("offline"))
            email = Mock()
            buzzer = Mock()
            layer = OutputAlertLayer(
                hooks={
                    "push": push,
                    "email": email,
                    "buzzer": buzzer,
                },
                database_path=Path(directory) / "alerts.sqlite3",
                frame_directory=Path(directory) / "frames",
            )
            layer.emit_alerts(
                [{
                    "severity": "critical",
                    "title": "Urgent",
                    "message": "Act now.",
                    "objects": [],
                }],
                frame=np.zeros((8, 8, 3), dtype=np.uint8),
            )

            push.assert_called_once()
            email.assert_called_once()
            buzzer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
