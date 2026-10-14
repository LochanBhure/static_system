"""Tests for local Ollama response parsing and reasoning fallback."""

from datetime import datetime, timezone
from io import BytesIO
import json
from unittest.mock import Mock, patch
import unittest

from reasoning_engine import OllamaReasoningClient, ReasoningEngine


VALID_RESULT = {
    "alert_level": "MEDIUM",
    "alert_type": "unusual_movement",
    "reason": "Measured movement differs from recent history.",
    "confidence": 0.72,
    "affected_object_ids": ["cam1:p1"],
}


class OllamaReasoningTests(unittest.TestCase):
    def test_ollama_response_is_parsed_and_validated(self):
        response = BytesIO(json.dumps({
            "message": {"content": json.dumps(VALID_RESULT)}
        }).encode())
        with patch("reasoning_engine.urlopen", return_value=response) as urlopen:
            result = OllamaReasoningClient(
                host="http://127.0.0.1:11434",
                model="phi3:mini",
            ).analyze({
                "scene_context": {"static_objects": []},
                "live_context": {
                    "moving_objects": [{"id": "cam1:p1"}],
                },
            })

        self.assertEqual(result, VALID_RESULT)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(json.loads(request.data)["model"], "phi3:mini")

    def test_invalid_confidence_is_rejected(self):
        result = dict(VALID_RESULT, confidence=1.5)
        with self.assertRaisesRegex(ValueError, "confidence"):
            OllamaReasoningClient.validate_result(result)

    def test_extra_response_field_is_rejected(self):
        result = dict(VALID_RESULT, extra="not allowed")
        with self.assertRaisesRegex(ValueError, "exactly"):
            OllamaReasoningClient.validate_result(result)

    def test_reasoning_engine_uses_llm_when_rules_are_quiet(self):
        client = Mock()
        client.analyze.return_value = VALID_RESULT
        engine = ReasoningEngine(llm_client=client)
        context = {
            "scene_context": {"static_objects": [], "restricted_zones": []},
            "live_context": {"moving_objects": [], "frame": 10, "fps": 30},
            "historical_context": {"frames": []},
        }

        alerts = engine.analyze(
            context,
            now=datetime(2026, 10, 14, 12, tzinfo=timezone.utc),
        )

        self.assertEqual(alerts, [VALID_RESULT])
        client.analyze.assert_called_once_with(context)

    def test_uncertain_rule_alert_also_consults_llm(self):
        client = Mock()
        client.analyze.return_value = dict(
            VALID_RESULT,
            alert_type="crowding_review",
            affected_object_ids=["p0", "p1", "p2", "p3", "p4"],
        )
        people = [
            {
                "id": f"p{index}",
                "class": "person",
                "camera_id": "cam1",
                "position": {
                    "x": 500 + index * 2,
                    "y": 500,
                    "nx": 0.5 + index * 0.002,
                    "ny": 0.5,
                },
            }
            for index in range(5)
        ]
        context = {
            "scene_context": {
                "static_objects": [],
                "restricted_zones": [],
            },
            "live_context": {
                "moving_objects": people,
                "frame": 10,
                "fps": 30,
            },
            "historical_context": {"frames": []},
        }

        alerts = ReasoningEngine(llm_client=client).analyze(
            context,
            now=datetime(2026, 10, 14, 12, tzinfo=timezone.utc),
        )

        self.assertEqual([alert["alert_type"] for alert in alerts], [
            "crowding",
            "crowding_review",
        ])
        client.analyze.assert_called_once_with(context)

    def test_none_result_suppresses_llm_alert(self):
        client = Mock()
        client.analyze.return_value = {
            "alert_level": "NONE",
            "alert_type": "",
            "reason": "No supported concern.",
            "confidence": 0.9,
            "affected_object_ids": [],
        }
        engine = ReasoningEngine(llm_client=client)
        result = engine.analyze({
            "scene_context": {"static_objects": [], "restricted_zones": []},
            "live_context": {"moving_objects": [], "frame": 1, "fps": 30},
            "historical_context": {"frames": []},
        })
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
