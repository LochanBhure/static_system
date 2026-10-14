"""Mock-provider tests proving email attachment and push payload behavior."""

from email.message import EmailMessage
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
import unittest
from urllib.error import URLError

import config
from providers.email_provider import send_email_alert
from providers.push_provider import send_push_alert


class ProviderTests(unittest.TestCase):
    def test_ntfy_request_contains_alert_title_message_and_priority(self):
        response = Mock()
        response.status = 200
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with (
            patch.object(config, "NTFY_TOPIC", "test-topic"),
            patch("providers.push_provider.urlopen", return_value=response) as call,
        ):
            send_push_alert(
                {"title": "Urgent", "message": "Check gate."},
                {"level": "HIGH"},
            )

        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://ntfy.sh/test-topic")
        self.assertEqual(json.loads(request.data), {
            "title": "Urgent",
            "message": "Check gate.",
            "priority": 5,
        })

    def test_ntfy_failure_uses_configured_pushover_fallback(self):
        with (
            patch.object(config, "PUSH_PROVIDER", "ntfy"),
            patch.object(config, "NTFY_TOPIC", "test-topic"),
            patch.object(config, "PUSHOVER_TOKEN", "app-token"),
            patch.object(config, "PUSHOVER_USER_KEY", "user-key"),
            patch(
                "providers.push_provider._send_ntfy",
                side_effect=URLError("network down"),
            ),
            patch("providers.push_provider._send_pushover") as fallback,
        ):
            send_push_alert(
                {"title": "Alert", "message": "Message"},
                {"level": "MEDIUM"},
            )
        fallback.assert_called_once()

    def test_high_email_attaches_saved_jpeg(self):
        smtp = Mock()
        smtp.__enter__ = Mock(return_value=smtp)
        smtp.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.jpg"
            image_path.write_bytes(b"jpeg-frame")
            with (
                patch.object(config, "SMTP_HOST", "smtp.example.test"),
                patch.object(config, "SMTP_PORT", 587),
                patch.object(config, "SMTP_USER", "camera@example.test"),
                patch.object(config, "SMTP_PASS", "test-password"),
                patch.object(config, "ALERT_EMAIL_TO", "security@example.test"),
                patch("providers.email_provider.smtplib.SMTP", return_value=smtp),
            ):
                send_email_alert(
                    {"title": "Urgent", "message": "Check gate."},
                    {
                        "level": "HIGH",
                        "snapshot_path": str(image_path),
                        "camera_id": "north",
                        "frame_number": 42,
                        "coordinates": {"north:p1": {"x": 10, "y": 20}},
                    },
                )

        smtp.starttls.assert_called_once()
        smtp.login.assert_called_once_with(
            "camera@example.test", "test-password"
        )
        message = smtp.send_message.call_args.args[0]
        self.assertIsInstance(message, EmailMessage)
        body = message.get_body(preferencelist=("plain",)).get_content()
        self.assertIn("Camera: north", body)
        self.assertIn("Frame: 42", body)
        attachments = list(message.iter_attachments())
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].get_filename(), "frame.jpg")
        self.assertEqual(attachments[0].get_payload(decode=True), b"jpeg-frame")


if __name__ == "__main__":
    unittest.main()
