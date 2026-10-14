"""Concurrent camera pipeline management and tagged coordinate merging."""

from copy import deepcopy
from datetime import datetime, timedelta
import threading
import time
import traceback

import config


class CameraManager:
    """Run an independent static/live pipeline for every configured camera."""

    def __init__(self, cameras=None, pipeline_factory=None):
        self.cameras = self._camera_configs(cameras)
        if not self.cameras:
            raise ValueError("At least one camera must be configured.")
        ids = [camera["camera_id"] for camera in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("Camera IDs must be unique.")
        self.pipeline_factory = pipeline_factory or self._default_pipeline_factory
        self.pipelines = {}
        try:
            for camera in self.cameras:
                self.pipelines[camera["camera_id"]] = self.pipeline_factory(
                    camera
                )
        except Exception:
            for pipeline in self.pipelines.values():
                pipeline.cap.release()
            raise
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._pipeline_locks = {
            camera_id: threading.Lock() for camera_id in self.pipelines
        }
        self._threads = []
        self._errors = {}
        self._last_presence = {camera_id: {} for camera_id in ids}
        self._recent_exits = []
        self._continuity_events = []

    @staticmethod
    def _camera_configs(cameras):
        source_cameras = cameras if cameras is not None else config.CAMERAS
        if not source_cameras:
            source_cameras = [{
                "camera_id": "camera_1",
                "source": config.CAMERA_SOURCE,
                "name": "Camera 1",
                "location": config.CAMERA_LOCATION,
                "restricted_zones": config.RESTRICTED_ZONES,
            }]
        normalized = []
        for index, camera in enumerate(source_cameras, start=1):
            source = camera.get("source", camera.get("url"))
            if source is None:
                raise ValueError(
                    f"Camera {index} must specify a source or url."
                )
            normalized.append({
                **camera,
                "camera_id": camera.get("camera_id", f"camera_{index}"),
                "source": source,
                "name": camera.get("name", f"Camera {index}"),
                "location": camera.get("location", "Unknown"),
                "restricted_zones": camera.get("restricted_zones", []),
            })
        return normalized

    @staticmethod
    def _default_pipeline_factory(camera):
        from static_environment import StaticEnvironmentSystem

        return StaticEnvironmentSystem(
            camera_config=camera,
            enable_reasoning=False,
            display=False,
        )

    def start(self):
        """Start all independent camera reader/processor threads."""
        if self._threads:
            raise RuntimeError("Camera manager has already been started.")
        self._stop_event.clear()
        for camera_id, pipeline in self.pipelines.items():
            thread = threading.Thread(
                target=self._process_camera,
                args=(camera_id, pipeline),
                name=f"camera-{camera_id}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def _process_camera(self, camera_id, pipeline):
        try:
            while not self._stop_event.is_set():
                success, frame = pipeline.cap.read()
                if not success:
                    break
                with self._pipeline_locks[camera_id]:
                    pipeline.frame_count += 1
                    pipeline.process_frame(frame)
        except Exception:
            with self._lock:
                self._errors[camera_id] = traceback.format_exc()
            self._stop_event.set()
        finally:
            pipeline.cap.release()

    def run(self, on_snapshot=None, interval_seconds=0.1, stop_event=None):
        """Start cameras and invoke callback with merged state until stopped."""
        self.start()
        try:
            while (
                not self._stop_event.is_set()
                and not (stop_event and stop_event.is_set())
            ):
                self._raise_worker_errors()
                if on_snapshot is not None:
                    on_snapshot(self.snapshot())
                if not any(thread.is_alive() for thread in self._threads):
                    break
                time.sleep(max(0.01, interval_seconds))
            self._raise_worker_errors()
        finally:
            self.stop()

    def snapshot(self):
        """Return camera-tagged static/live coordinates and continuity events."""
        now = datetime.now().astimezone()
        static_objects = []
        moving_objects = []
        history = {}
        cameras = []
        current_presence = {}
        restricted_zones = []

        for camera in self.cameras:
            camera_id = camera["camera_id"]
            pipeline = self.pipelines[camera_id]
            cameras.append({
                "camera_id": camera_id,
                "name": camera["name"],
                "location": camera["location"],
            })
            restricted_zones.extend(
                {**zone, "camera_id": camera_id}
                for zone in camera["restricted_zones"]
            )
            with self._pipeline_locks[camera_id]:
                for obj in pipeline.static_objects.values():
                    tagged = self._tag_object(obj, camera_id)
                    tagged["camera_frame"] = pipeline.frame_count
                    tagged["camera_fps"] = pipeline.fps
                    static_objects.append(tagged)
                for obj in pipeline.current_live_objects:
                    tagged = self._tag_object(obj, camera_id)
                    tagged["camera_frame"] = pipeline.frame_count
                    tagged["camera_fps"] = pipeline.fps
                    moving_objects.append(tagged)
                    if obj.get("class") == "person":
                        current_presence[(camera_id, obj["id"])] = {
                            "camera_id": camera_id,
                            "object_id": obj["id"],
                            "class": "person",
                            "position": deepcopy(obj.get("position", {})),
                        }
                history[camera_id] = deepcopy(list(pipeline.frame_history))

        self._update_cross_camera_events(current_presence, now)
        return {
            "timestamp": now.isoformat(),
            "cameras": cameras,
            "static_objects": static_objects,
            "moving_objects": moving_objects,
            "restricted_zones": restricted_zones,
            "history_by_camera": history,
            "cross_camera_events": list(self._continuity_events),
        }

    def latest_frame_context(self, camera_id):
        """Return a safe copy of the most recent frame and its frame number."""
        if camera_id not in self.pipelines:
            raise KeyError(f"Unknown camera ID: {camera_id}")
        pipeline = self.pipelines[camera_id]
        with self._pipeline_locks[camera_id]:
            frame = (
                pipeline.latest_frame.copy()
                if pipeline.latest_frame is not None else None
            )
            return frame, pipeline.frame_count

    @staticmethod
    def _tag_object(obj, camera_id):
        tagged = deepcopy(obj)
        tagged["source_object_id"] = tagged.get("id")
        tagged["id"] = f"{camera_id}:{tagged.get('id', 'unknown')}"
        tagged["camera_id"] = camera_id
        return tagged

    def _update_cross_camera_events(self, current_presence, now):
        active_by_camera = {
            camera_id: set() for camera_id in self.pipelines
        }
        for camera_id, object_id in current_presence:
            active_by_camera[camera_id].add(("person", object_id))

        for camera_id, current_ids in active_by_camera.items():
            previous_ids = self._last_presence[camera_id]
            for object_key in previous_ids.keys() - current_ids:
                previous = previous_ids[object_key]
                self._recent_exits.append({
                    **previous,
                    "exited_at": now,
                })

        for camera_id, object_id in current_presence:
            presence_key = ("person", object_id)
            if presence_key in self._last_presence[camera_id]:
                continue
            for exit_record in reversed(self._recent_exits):
                age = (now - exit_record["exited_at"]).total_seconds()
                if (
                    exit_record["camera_id"] != camera_id
                    and age <= config.CROSS_CAMERA_WINDOW_SECONDS
                ):
                    self._continuity_events.append({
                        "type": "possible_cross_camera_continuity",
                        "from_camera_id": exit_record["camera_id"],
                        "from_object_id": exit_record["object_id"],
                        "to_camera_id": camera_id,
                        "to_object_id": object_id,
                        "seconds_between": round(age, 2),
                        "confidence": 0.0,
                        "created_at": now.isoformat(),
                        "evidence": (
                            "Person tracks disappeared from one camera and "
                            "appeared on another within the configured time "
                            "window; identity is not confirmed."
                        ),
                    })
                    self._recent_exits.remove(exit_record)
                    break

        self._last_presence = {
            camera_id: {
                object_key: current_presence[(camera_id, object_key[1])]
                for object_key in object_keys
                if (camera_id, object_key[1]) in current_presence
            }
            for camera_id, object_keys in active_by_camera.items()
        }
        cutoff = now - timedelta(seconds=config.CROSS_CAMERA_WINDOW_SECONDS)
        self._recent_exits = [
            record for record in self._recent_exits
            if record["exited_at"] >= cutoff
        ]
        self._continuity_events = [
            event for event in self._continuity_events
            if (
                now - datetime.fromisoformat(event["created_at"])
            ).total_seconds() <= config.CROSS_CAMERA_WINDOW_SECONDS
        ]

    def _raise_worker_errors(self):
        with self._lock:
            errors = dict(self._errors)
        if errors:
            details = "\n".join(
                f"{camera_id}:\n{trace}" for camera_id, trace in errors.items()
            )
            raise RuntimeError(f"Camera processing failed:\n{details}")

    def stop(self):
        """Stop worker loops, wait for threads, and release camera resources."""
        self._stop_event.set()
        for pipeline in self.pipelines.values():
            pipeline.cap.release()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()
