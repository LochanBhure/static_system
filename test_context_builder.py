"""Tests for the normalized three-layer reasoning context."""

import unittest

from context_builder import ContextBuilder


class ContextBuilderTests(unittest.TestCase):
    def test_context_contains_scene_live_history_and_events(self):
        builder = ContextBuilder()
        history = [{
            "frame": 9,
            "timestamp": "2026-10-14T09:00:00+05:30",
            "moving_objects": [{
                "id": "person_1",
                "class": "person",
                "position": {"x": 100, "y": 90, "nx": 0.5, "ny": 0.45},
                "velocity": {"vx": 3.0, "vy": 4.0, "vz": 0.0},
                "direction": "east",
            }],
        }]
        current = [{
            "id": "person_1",
            "class": "person",
            "position": {"x": 110, "y": 90, "nx": 0.55, "ny": 0.45},
            "velocity": 1.5,
            "direction": "east",
        }]
        context = builder.build_context(
            {
                "frame": 10,
                "static_objects": [{
                    "id": "door",
                    "class": "door",
                    "coordinates": {"x": 200, "y": 100, "nx": 0.8, "ny": 0.5},
                }],
            },
            current,
            history,
            camera_id="north",
            camera_name="North entrance",
            camera_location="Entrance",
            restricted_zones=[{
                "name": "restricted",
                "bbox": [0.7, 0.0, 1.0, 1.0],
            }],
        )

        self.assertEqual(context["scene_context"]["camera_name"], "North entrance")
        self.assertEqual(
            context["scene_context"]["static_objects"][0]["id"], "door"
        )
        self.assertEqual(
            context["scene_context"]["restricted_zones"][0]["name"],
            "restricted",
        )
        self.assertEqual(context["live_context"]["frame"], 10)
        self.assertEqual(
            context["live_context"]["moving_objects"][0]["id"], "person_1"
        )
        self.assertEqual(
            context["historical_context"]["window_frames"], 1
        )
        self.assertEqual(
            context["historical_context"]["movement_patterns"][0]["directions"],
            {"east": 1},
        )
        self.assertEqual(
            context["historical_context"]["movement_patterns"][0]["max_velocity"],
            5.0,
        )
        self.assertEqual(
            context["historical_context"]["entry_exit_events"][0]["type"],
            "entry",
        )
        self.assertTrue(
            context["historical_context"]["entry_exit_events"][0]["timestamp"]
        )

    def test_invalid_scene_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            ContextBuilder().build_context([], [], [])


if __name__ == "__main__":
    unittest.main()
