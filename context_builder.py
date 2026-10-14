"""Build compact scene, live, and historical context for local reasoning."""

import json
from collections import Counter, defaultdict
from datetime import datetime


class ContextBuilder:
    """Normalize the static cache and frame history for a text LLM."""

    def build_context(
        self,
        scene,
        live_objects,
        history,
        *,
        camera_id=None,
        camera_name=None,
        camera_location=None,
        restricted_zones=None,
        timestamp=None,
    ):
        scene_data = json.loads(scene) if isinstance(scene, str) else scene
        if not isinstance(scene_data, dict):
            raise ValueError("Scene context must be a JSON object.")

        static_objects = self._object_list(
            scene_data.get("static_objects", scene_data.get("objects", []))
        )
        live_objects = self._object_list(live_objects)
        history_frames = list(history)
        current_timestamp = timestamp or datetime.now().astimezone().isoformat()
        frame_number = scene_data.get("frame", scene_data.get("frame_number"))

        zones = (
            restricted_zones
            if restricted_zones is not None
            else scene_data.get("restricted_zones", [])
        )
        if not isinstance(zones, list):
            raise ValueError("Restricted zones must be a list.")

        movement_patterns = self._summarize_movement(history_frames, live_objects)
        entry_exit_events = self._entry_exit_events(
            history_frames,
            live_objects,
            current_timestamp,
        )

        return {
            "camera_id": camera_id or scene_data.get("camera_id"),
            "scene_context": {
                "camera_id": camera_id or scene_data.get("camera_id"),
                "camera_name": camera_name or scene_data.get("camera_name"),
                "camera_location": (
                    camera_location
                    or scene_data.get("camera_location")
                    or "Unknown"
                ),
                "static_objects": static_objects,
                "restricted_zones": zones,
            },
            "live_context": {
                "frame": frame_number,
                "timestamp": current_timestamp,
                "moving_objects": live_objects,
            },
            "historical_context": {
                "window_frames": len(history_frames),
                "frames": history_frames,
                "movement_patterns": movement_patterns,
                "entry_exit_events": entry_exit_events,
            },
        }

    @staticmethod
    def _object_list(objects):
        if objects is None:
            return []
        if isinstance(objects, dict):
            if "id" in objects:
                return [objects]
            return list(objects.values())
        if isinstance(objects, list):
            if not all(isinstance(obj, dict) for obj in objects):
                raise ValueError("Object collections must contain JSON objects.")
            return objects
        raise ValueError("Object collections must be a list or object map.")

    @staticmethod
    def _objects_in_frame(frame):
        if not isinstance(frame, dict):
            raise ValueError("History frames must be JSON objects.")
        return ContextBuilder._object_list(
            frame.get(
                "moving_objects",
                frame.get("live_objects", frame.get("objects", [])),
            )
        )

    def _summarize_movement(self, history, current_objects):
        by_object = defaultdict(list)
        for frame in history:
            frame_number = frame.get("frame")
            for obj in self._objects_in_frame(frame):
                position = obj.get("position", obj)
                by_object[obj.get("id", "unknown")].append({
                    "frame": frame_number,
                    "x": position.get("x"),
                    "y": position.get("y"),
                    "direction": obj.get("direction", "unknown"),
                    "velocity": self._velocity_value(obj),
                })

        patterns = []
        for object_id, observations in by_object.items():
            directions = Counter(
                item["direction"]
                for item in observations
                if item["direction"] not in ("unknown", "none")
            )
            speeds = [
                self._velocity_value(item)
                for item in observations
            ]
            patterns.append({
                "object_id": object_id,
                "observations": len(observations),
                "directions": dict(directions),
                "max_velocity": max(speeds, default=0.0),
                "start_position": self._position(observations[0]),
                "end_position": self._position(observations[-1]),
            })

        known_ids = {pattern["object_id"] for pattern in patterns}
        for obj in current_objects:
            object_id = obj.get("id", "unknown")
            if object_id in known_ids:
                continue
            position = obj.get("position", obj)
            patterns.append({
                "object_id": object_id,
                "observations": 1,
                "directions": (
                    {obj["direction"]: 1}
                    if obj.get("direction") not in (None, "unknown", "none")
                    else {}
                ),
                "max_velocity": self._velocity_value(obj),
                "start_position": {
                    "x": position.get("x"),
                    "y": position.get("y"),
                },
                "end_position": {
                    "x": position.get("x"),
                    "y": position.get("y"),
                },
            })
        return patterns

    @staticmethod
    def _position(observation):
        return {"x": observation["x"], "y": observation["y"]}

    @staticmethod
    def _velocity_value(obj):
        scalar = obj.get("speed_body_lengths_per_second")
        if isinstance(scalar, (int, float)):
            return float(scalar)
        velocity = obj.get("velocity", 0.0)
        if isinstance(velocity, dict):
            components = [
                float(velocity.get(axis, 0.0))
                for axis in ("vx", "vy", "vz")
            ]
            return float(sum(component ** 2 for component in components) ** 0.5)
        return float(velocity) if isinstance(velocity, (int, float)) else 0.0

    def _entry_exit_events(self, history, current_objects, timestamp):
        events = []
        previous_ids = set()
        for frame in history:
            frame_objects = self._objects_in_frame(frame)
            current_ids = {
                obj["id"] for obj in frame_objects if isinstance(obj.get("id"), str)
            }
            frame_time = frame.get("timestamp") or timestamp
            for object_id in current_ids - previous_ids:
                events.append({
                    "type": "entry",
                    "object_id": object_id,
                    "timestamp": frame_time,
                    "frame": frame.get("frame"),
                })
            for object_id in previous_ids - current_ids:
                events.append({
                    "type": "exit",
                    "object_id": object_id,
                    "timestamp": frame_time,
                    "frame": frame.get("frame"),
                })
            previous_ids = current_ids

        current_ids = {
            obj["id"] for obj in current_objects if isinstance(obj.get("id"), str)
        }
        for object_id in current_ids - previous_ids:
            events.append({
                "type": "entry",
                "object_id": object_id,
                "timestamp": timestamp,
                "frame": None,
            })
        for object_id in previous_ids - current_ids:
            events.append({
                "type": "exit",
                "object_id": object_id,
                "timestamp": timestamp,
                "frame": None,
            })
        return events
