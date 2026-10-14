"""Tests for multi-camera coordinate tagging and handoff candidates."""

from collections import deque
from types import SimpleNamespace
import unittest

from camera_manager import CameraManager
from context_builder import ContextBuilder
from main import _build_context
from static_environment import StaticEnvironmentSystem


class FakePipeline:
    def __init__(self, live_objects=None):
        self.static_objects = {}
        self.current_live_objects = live_objects or []
        self.frame_history = deque()
        self.latest_frame = None
        self.frame_count = 0
        self.fps = 30.0
        self.cap = SimpleNamespace(release=lambda: None)


def observed_person(object_id):
    return {
        "id": object_id,
        "class": "person",
        "position": {"x": 120, "y": 80, "nx": 0.4, "ny": 0.3},
        "confidence": 0.9,
    }


class CameraManagerTests(unittest.TestCase):
    def setUp(self):
        self.pipelines = {
            "north": FakePipeline([observed_person("Person1")]),
            "south": FakePipeline(),
        }
        self.pipelines["north"].static_objects["static_1"] = {
            "id": "static_1",
            "class": "door",
            "coordinates": {"x": 200, "y": 100, "nx": 0.8, "ny": 0.5},
        }
        cameras = [
            {
                "camera_id": "north",
                "source": 0,
                "name": "North door",
                "location": "North entrance",
                "restricted_zones": [{
                    "name": "secure",
                    "bbox": [0.7, 0.0, 1.0, 1.0],
                }],
            },
            {
                "camera_id": "south",
                "source": 1,
                "name": "South door",
                "location": "South entrance",
                "restricted_zones": [],
            },
        ]
        self.manager = CameraManager(
            cameras=cameras,
            pipeline_factory=lambda camera: self.pipelines[
                camera["camera_id"]
            ],
        )

    def test_snapshot_tags_each_camera_and_merges_coordinates(self):
        snapshot = self.manager.snapshot()
        self.assertEqual(len(snapshot["cameras"]), 2)
        self.assertEqual(
            snapshot["moving_objects"][0]["id"], "north:Person1"
        )
        self.assertEqual(
            snapshot["moving_objects"][0]["camera_id"], "north"
        )
        self.assertEqual(snapshot["static_objects"][0]["id"], "north:static_1")
        self.assertEqual(snapshot["restricted_zones"][0]["camera_id"], "north")

    def test_camera_exit_and_entry_create_uncertain_handoff_event(self):
        self.manager.snapshot()
        self.pipelines["north"].current_live_objects = []
        self.pipelines["south"].current_live_objects = [
            observed_person("Person2")
        ]

        snapshot = self.manager.snapshot()

        event = snapshot["cross_camera_events"][0]
        self.assertEqual(event["from_camera_id"], "north")
        self.assertEqual(event["to_camera_id"], "south")
        self.assertEqual(event["confidence"], 0.0)
        self.assertIn("identity is not confirmed", event["evidence"])

    def test_merged_context_retains_camera_tags_and_cross_events(self):
        snapshot = self.manager.snapshot()
        snapshot["cross_camera_events"] = [{"type": "possible_cross_camera_continuity"}]

        context = _build_context(ContextBuilder(), snapshot)

        self.assertEqual(
            context["live_context"]["moving_objects"][0]["id"],
            "north:Person1",
        )
        self.assertEqual(
            context["scene_context"]["static_objects"][0]["camera_id"],
            "north",
        )
        self.assertEqual(
            context["scene_context"]["restricted_zones"][0]["camera_id"],
            "north",
        )
        self.assertEqual(
            context["historical_context"]["cross_camera_events"],
            snapshot["cross_camera_events"],
        )

    def test_live_pipeline_emits_depth_and_velocity_vector(self):
        pipeline = StaticEnvironmentSystem.__new__(StaticEnvironmentSystem)
        pipeline.frame_count = 1
        pipeline.fps = 30.0
        pipeline.live_tracks = {}
        pipeline.next_person_id = 1
        pipeline.next_live_track_id = 1
        pipeline.recent_behavior_events = deque(maxlen=10)
        pipeline.restricted_zones = []
        first = {
            "class": "person",
            "confidence": 0.9,
            "bbox": [100.0, 100.0, 140.0, 200.0],
        }
        pipeline.update_live_tracks([first], (480, 640, 3))

        pipeline.frame_count = 31
        moved = {
            "class": "person",
            "confidence": 0.9,
            "bbox": [130.0, 100.0, 170.0, 200.0],
        }
        live = pipeline.update_live_tracks([moved], (480, 640, 3))

        self.assertGreater(live[0]["position"]["z"], 0)
        self.assertEqual(live[0]["velocity"]["vx"], 30.0)
        self.assertEqual(live[0]["velocity"]["vy"], 0.0)
        self.assertEqual(live[0]["velocity"]["vz"], 0.0)


if __name__ == "__main__":
    unittest.main()
