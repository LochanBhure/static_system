"""Generate, persist, and optionally dispatch periodic shift reports."""

import json
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

import config
from db.database import Database


class ShiftReportGenerator:
    """Create a structured human-readable report from persisted alerts."""

    def __init__(
        self,
        database=None,
        *,
        host=None,
        model=None,
        timeout=None,
        report_hooks=None,
    ):
        self.database = database or Database()
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self.report_hooks = list(
            config.SHIFT_REPORT_HOOKS if report_hooks is None else report_hooks
        )
        if not all(callable(hook) for hook in self.report_hooks):
            raise TypeError("Every shift report hook must be callable.")

    def generate_report(self, period_start, period_end):
        """Summarize alert rows during [period_start, period_end)."""
        if period_start.tzinfo is None or period_end.tzinfo is None:
            raise ValueError("Shift report boundaries must be timezone-aware.")
        if period_end <= period_start:
            raise ValueError("period_end must be later than period_start.")

        alerts = self.database.get_alerts(period_start, period_end)
        generated = self._summarize(alerts)
        report = {
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "total_events": len(alerts),
            "high_alerts": [
                self._alert_summary(alert) for alert in alerts
                if alert["level"] == "HIGH"
            ],
            "medium_alerts": [
                self._alert_summary(alert) for alert in alerts
                if alert["level"] == "MEDIUM"
            ],
            "summary": generated["summary"],
            "anomalies": generated["anomalies"],
        }
        self.database.save_report(report)
        for hook in self.report_hooks:
            hook(report)
        return report

    def generate_last_interval(self, period_end=None):
        end = period_end or datetime.now().astimezone()
        start = end - timedelta(minutes=config.SHIFT_REPORT_INTERVAL_MINUTES)
        return self.generate_report(start, end)

    def _summarize(self, alerts):
        payload = {
            "alerts": [
                {
                    "created_at": alert["created_at"],
                    "camera_id": alert["camera_id"],
                    "level": alert["level"],
                    "title": alert["title"],
                    "message": alert["message"],
                    "objects": alert["objects"],
                    "coordinates": alert["coordinates"],
                }
                for alert in alerts
            ]
        }
        prompt = (
            "Summarize this local CCTV alert data. Do not infer intent or "
            "facts beyond the records. Return only JSON with exactly "
            '"summary" (string) and "anomalies" (array of strings). '
            "Alert records: "
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        )
        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps({
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You write concise CCTV shift reports using only "
                            "the supplied alert records."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        if not isinstance(body, dict):
            raise ValueError("Ollama report response must be a JSON object.")
        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), str
        ):
            raise ValueError("Ollama report response is missing message content.")
        content = message["content"]
        result = json.loads(content)
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("summary"), str)
            or not isinstance(result.get("anomalies"), list)
            or not all(
                isinstance(item, str) for item in result["anomalies"]
            )
        ):
            raise ValueError(
                "Ollama report must contain summary text and string anomalies."
            )
        return result

    @staticmethod
    def _alert_summary(alert):
        return {
            "created_at": alert["created_at"],
            "camera_id": alert["camera_id"],
            "title": alert["title"],
            "message": alert["message"],
            "objects": alert["objects"],
            "coordinates": alert["coordinates"],
            "snapshot_path": alert["snapshot_path"],
        }
