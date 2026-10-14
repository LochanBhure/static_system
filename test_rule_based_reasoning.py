"""Mock-coordinate tests for deterministic security rules."""

from datetime import datetime, timedelta, timezone
import math
import unittest

from reasoning_engine import RuleBasedReasoningEngine


def person(object_id, x=0.5, y=0.5, *, velocity=0.0, camera_id="cam1"):
    return {
        "id": object_id,
        "class": "person",
        "camera_id": camera_id,
        "position": {
            "x": x * 1000,
            "y": y * 1000,
            "nx": x,
            "ny": y,
        },
        "body_height_px": 100,
        "velocity": velocity,
    }


def context(*, moving=None, static=None, zones=None, history=None, frame=300):
    return {
        "scene_context": {
            "static_objects": static or [],
            "restricted_zones": zones or [],
        },
        "live_context": {
            "frame": frame,
            "fps": 30,
            "moving_objects": moving or [],
        },
        "historical_context": {"frames": history or []},
    }


class RuleBasedReasoningTests(unittest.TestCase):
    daytime = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)

    def test_person_inside_restricted_zone_is_high(self):
        alerts = RuleBasedReasoningEngine().evaluate(
            context(
                moving=[person("p1", 0.8, 0.5)],
                zones=[{
                    "name": "secure",
                    "camera_id": "cam1",
                    "bbox": [0.7, 0.2, 0.9, 0.8],
                }],
            ),
            now=self.daytime,
        )
        self.assertEqual(alerts[0]["alert_type"], "restricted_zone_entry")
        self.assertEqual(alerts[0]["alert_level"], "HIGH")

    def test_unattended_static_package_is_medium(self):
        package = {
            "id": "bag1",
            "class": "backpack",
            "camera_id": "cam1",
            "coordinates": {"x": 300, "y": 400, "nx": 0.3, "ny": 0.4},
            "scene_info": {"first_seen_frame": 0},
        }
        alerts = RuleBasedReasoningEngine().evaluate(
            context(static=[package], frame=1200),
            now=self.daytime,
        )
        self.assertEqual(alerts[0]["alert_type"], "possibly_unattended_item")
        self.assertEqual(alerts[0]["alert_level"], "MEDIUM")

    def test_rapid_convergence_is_high(self):
        history = [{
            "frame": 30,
            "moving_objects": [
                person("p1", 0.1, 0.5),
                person("p2", 0.5, 0.5),
            ],
        }]
        alerts = RuleBasedReasoningEngine().evaluate(
            context(
                moving=[
                    person("p1", 0.4, 0.5),
                    person("p2", 0.47, 0.5),
                ],
                history=history,
                frame=60,
            ),
            now=self.daytime,
        )
        self.assertIn("rapid_approach", [alert["alert_type"] for alert in alerts])
        self.assertEqual(
            next(
                alert for alert in alerts
                if alert["alert_type"] == "rapid_approach"
            )["alert_level"],
            "HIGH",
        )

    def test_loitering_in_small_area_is_medium(self):
        engine = RuleBasedReasoningEngine()
        alerts = []
        for seconds in range(0, 31):
            alerts = engine.evaluate(
                context(moving=[person("p1", 0.5, 0.5)]),
                now=self.daytime + timedelta(seconds=seconds),
            )
        self.assertIn("possible_loitering", [a["alert_type"] for a in alerts])

    def test_circling_within_loiter_radius_is_detected(self):
        engine = RuleBasedReasoningEngine()
        alerts = []
        for seconds in range(31):
            angle = 2 * math.pi * seconds / 30
            current_person = person(
                "p1",
                0.5 + 0.04 * math.cos(angle),
                0.5 + 0.04 * math.sin(angle),
            )
            alerts = engine.evaluate(
                context(moving=[current_person]),
                now=self.daytime + timedelta(seconds=seconds),
            )
        self.assertIn("possible_loitering", [a["alert_type"] for a in alerts])

    def test_high_velocity_is_medium(self):
        alerts = RuleBasedReasoningEngine().evaluate(
            context(moving=[person("p1", velocity=6.0)]),
            now=self.daytime,
        )
        self.assertIn("high_velocity", [a["alert_type"] for a in alerts])

    def test_clustered_people_is_low(self):
        people = [
            person(f"p{index}", 0.5 + index * 0.005, 0.5)
            for index in range(5)
        ]
        alerts = RuleBasedReasoningEngine().evaluate(
            context(moving=people),
            now=self.daytime,
        )
        crowd = next(a for a in alerts if a["alert_type"] == "crowding")
        self.assertEqual(crowd["alert_level"], "LOW")
        self.assertEqual(len(crowd["affected_object_ids"]), 5)

    def test_after_hours_activity_is_high(self):
        alerts = RuleBasedReasoningEngine().evaluate(
            context(moving=[person("p1")]),
            now=datetime(2026, 10, 14, 23, tzinfo=timezone.utc),
        )
        self.assertIn("after_hours_activity", [a["alert_type"] for a in alerts])

    def test_active_condition_emits_only_once_until_cleared(self):
        engine = RuleBasedReasoningEngine()
        restricted_context = context(
            moving=[person("p1", 0.8, 0.5)],
            zones=[{"bbox": [0.7, 0.0, 1.0, 1.0]}],
        )
        self.assertEqual(len(engine.evaluate(restricted_context, now=self.daytime)), 1)
        self.assertEqual(len(engine.evaluate(restricted_context, now=self.daytime)), 0)
        engine.evaluate(context(), now=self.daytime)
        self.assertEqual(len(engine.evaluate(restricted_context, now=self.daytime)), 1)


if __name__ == "__main__":
    unittest.main()
