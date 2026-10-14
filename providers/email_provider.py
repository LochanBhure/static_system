"""SMTP email delivery for CCTV alerts."""

import json
import smtplib
from email.message import EmailMessage
from pathlib import Path
import ssl

import config


def send_email_alert(alert, context):
    """Send an alert by SMTP and attach the HIGH-alert snapshot."""
    required = {
        "SMTP_HOST": config.SMTP_HOST,
        "SMTP_USER": config.SMTP_USER,
        "SMTP_PASS": config.SMTP_PASS,
        "ALERT_EMAIL_TO": config.ALERT_EMAIL_TO,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Missing email configuration: " + ", ".join(missing)
        )

    title = alert.get("title") or alert.get("alert_type", "CCTV alert")
    reason = alert.get("message") or alert.get("reason", "")
    details = [
        str(reason),
        f"Camera: {context.get('camera_id') or 'unknown'}",
        f"Frame: {context.get('frame_number') or 'unknown'}",
        "Coordinates: "
        + json.dumps(context.get("coordinates", {}), sort_keys=True),
    ]
    message = EmailMessage()
    message["Subject"] = f"[{context.get('level', 'ALERT')}] {title}"
    message["From"] = config.SMTP_USER
    message["To"] = config.ALERT_EMAIL_TO
    message.set_content("\n".join(details))

    snapshot_path = context.get("snapshot_path")
    if context.get("level") == "HIGH":
        if not snapshot_path:
            raise ValueError("HIGH alert email requires a saved frame.")
        image_path = Path(snapshot_path)
        image_data = image_path.read_bytes()
        message.add_attachment(
            image_data,
            maintype="image",
            subtype="jpeg",
            filename=image_path.name,
        )

    smtp_port = int(config.SMTP_PORT)
    if smtp_port == 465:
        with smtplib.SMTP_SSL(
            config.SMTP_HOST,
            smtp_port,
            context=ssl.create_default_context(),
            timeout=config.OLLAMA_TIMEOUT,
        ) as smtp:
            smtp.login(config.SMTP_USER, config.SMTP_PASS)
            smtp.send_message(message)
        return

    with smtplib.SMTP(
        config.SMTP_HOST,
        smtp_port,
        timeout=config.OLLAMA_TIMEOUT,
    ) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(config.SMTP_USER, config.SMTP_PASS)
        smtp.send_message(message)
