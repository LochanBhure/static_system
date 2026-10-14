"""Context, Ollama reasoning, and console alert layers."""

import json
import smtplib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

import config
from db.database import Database


class ContextLayer:
    """Build a compact scene context from the merged coordinate JSON."""

    def build_context(self, merged_json, live_objects, history):
        scene = json.loads(merged_json)
        objects = list(scene.get("objects", {}).values())

        classes = Counter(obj["class"] for obj in objects)
        zones = defaultdict(list)
        for obj in objects:
            zones[obj["scene_info"]["zone"]].append(obj["id"])

        return {
            "scene_context": {
                "camera_location": scene["camera_location"],
                "static_objects": objects,
                "class_counts": dict(classes),
                "occupied_zones": dict(zones),
                "restricted_zones": scene["restricted_zones"],
            },
            "live_context": {
                "frame": scene["frame"],
                "timestamp": scene["timestamp"],
                "moving_objects": live_objects,
            },
            "historical_context": {
                "window_frames": len(history),
                "frames": history,
            },
        }


class RuleBasedReasoningLayer:
    """Evaluate deterministic zone, unattended-item, and approach rules."""

    def __init__(self):
        self.active_alerts = set()
        self.previous_pair_distances = {}
        self.alerted_pairs = set()

    def evaluate(self, scene, live_objects, frame, fps):
        alerts = []
        current_conditions = set()
        people = [obj for obj in live_objects if obj["class"] == "person"]

        for person in people:
            for zone_name in person["restricted_zones"]:
                condition = ("restricted_zone", person["id"], zone_name)
                current_conditions.add(condition)
                if condition not in self.active_alerts:
                    alerts.append({
                        "severity": "warning",
                        "title": "Restricted-zone entry",
                        "message": (
                            f"{person['id']} entered restricted zone "
                            f"{zone_name} at normalized coordinates "
                            f"({person['position']['nx']}, "
                            f"{person['position']['ny']})."
                        ),
                        "objects": [person["id"]],
                    })

        for object_id, obj in scene["objects"].items():
            if obj["class"] not in config.ABANDONED_ITEM_CLASSES:
                continue
            elapsed = (
                frame - obj["scene_info"]["first_seen_frame"]
            ) / fps
            item_position = (
                obj["coordinates"]["nx"],
                obj["coordinates"]["ny"],
            )
            attended = any(
                np.hypot(
                    person["position"]["nx"] - item_position[0],
                    person["position"]["ny"] - item_position[1],
                ) <= config.PERSON_ITEM_DISTANCE_NORMALIZED
                for person in people
            )
            condition = ("abandoned_item", object_id)
            if elapsed >= config.ABANDONED_ITEM_SECONDS and not attended:
                current_conditions.add(condition)
                if condition not in self.active_alerts:
                    alerts.append({
                        "severity": "warning",
                        "title": "Possibly unattended item",
                        "message": (
                            f"{obj['class']} ({object_id}) has been detected "
                            f"for {elapsed:.1f} seconds without a person nearby."
                        ),
                        "objects": [object_id],
                    })

        current_pair_distances = {}
        for index, first in enumerate(people):
            for second in people[index + 1:]:
                pair = tuple(sorted((first["id"], second["id"])))
                first_position = first["position"]
                second_position = second["position"]
                separation = float(np.hypot(
                    first_position["x"] - second_position["x"],
                    first_position["y"] - second_position["y"],
                ))
                mean_body_height = max(
                    (first["body_height_px"] + second["body_height_px"]) / 2,
                    1.0,
                )
                separation_body_lengths = separation / mean_body_height
                current_pair_distances[pair] = (
                    separation_body_lengths,
                    frame,
                )

                previous = self.previous_pair_distances.get(pair)
                if previous is None:
                    continue
                previous_distance, previous_frame = previous
                elapsed_seconds = (frame - previous_frame) / fps
                if elapsed_seconds <= 0:
                    continue
                closing_speed = (
                    previous_distance - separation_body_lengths
                ) / elapsed_seconds
                close_and_fast = (
                    separation_body_lengths
                    <= config.RAPID_APPROACH_DISTANCE_BODY_LENGTHS
                    and closing_speed
                    >= config.RAPID_APPROACH_SPEED_BODY_LENGTHS_PER_SECOND
                )
                condition = ("rapid_approach", *pair)
                if close_and_fast:
                    current_conditions.add(condition)
                    if condition not in self.alerted_pairs:
                        alerts.append({
                            "severity": "warning",
                            "title": "Rapid approach - possible confrontation",
                            "message": (
                                f"{pair[0]} and {pair[1]} rapidly closed to "
                                f"{separation_body_lengths:.2f} body lengths "
                                f"apart at {closing_speed:.2f} body lengths "
                                "per second. This indicates proximity and "
                                "speed only, not confirmed fighting."
                            ),
                            "objects": list(pair),
                        })
                        self.alerted_pairs.add(condition)
                else:
                    self.alerted_pairs.discard(condition)

        self.previous_pair_distances = current_pair_distances
        self.alerted_pairs.intersection_update(current_conditions)
        self.active_alerts = current_conditions
        return alerts


class LLMReasoningLayer:
    """Ask an Ollama chat model to reason about the current scene context."""

    def __init__(self, base_url, model, timeout):
        self.endpoint = f"{base_url.rstrip('/')}/api/chat"
        self.model = model
        self.timeout = timeout

    def analyze(self, context):
        system_prompt = (
            "Analyze the supplied scene context, live moving objects, and "
            "recent frame history. Use restricted zones and entry durations "
            "when relevant. Distinguish moving from idle people and mention "
            "their IDs, positions, directions, and time since first seen "
            "(not necessarily actual entry time) when useful. Treat missing "
            "history as unknown. Compare historical locomotion and speed for "
            "abrupt walking-to-running transitions. Describe a detected "
            "speed increase as an unusual behavior change, not proof of "
            "suspicious intent, and include its measured evidence. If "
            "historical_context.behavior_events contains a "
            "sudden_speed_increase, include a warning alert describing the "
            "walking-to-running change and measured speeds. Compare position "
            "and direction changes across frames for repeated circular "
            "movement, and describe it as unusual movement rather than "
            "inferring intent. Treat historical_context.rule_alerts as "
            "deterministic rule results; do not contradict them. Only "
            "report events or hazards supported by "
            "the supplied data. Return JSON "
            "with exactly these keys: "
            '"summary" (string) and "alerts" (array). Each alert must have '
            '"severity" ("info", "warning", or "critical"), "title" (string), '
            '"message" (string), and "objects" (array of object IDs). Return '
            "an empty alerts array when there is nothing notable."
        )
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(context, separators=(",", ":")),
                },
            ],
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=self.timeout) as response:
            ollama_response = json.load(response)

        if not isinstance(ollama_response, dict):
            raise ValueError("Ollama API response must be a JSON object.")
        message = ollama_response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Ollama API response is missing message content.")

        result = json.loads(message["content"])
        self._validate_result(result)
        return result

    @staticmethod
    def _validate_result(result):
        if not isinstance(result, dict):
            raise ValueError("Ollama response must be a JSON object.")
        if not isinstance(result.get("summary"), str):
            raise ValueError("Ollama response must contain a string summary.")

        alerts = result.get("alerts")
        if not isinstance(alerts, list):
            raise ValueError("Ollama response must contain an alerts array.")

        valid_severities = {"info", "warning", "critical"}
        for alert in alerts:
            if not isinstance(alert, dict):
                raise ValueError("Each alert must be a JSON object.")
            if alert.get("severity") not in valid_severities:
                raise ValueError("Alert severity must be info, warning, or critical.")
            if not isinstance(alert.get("title"), str):
                raise ValueError("Each alert must contain a string title.")
            if not isinstance(alert.get("message"), str):
                raise ValueError("Each alert must contain a string message.")
            if not isinstance(alert.get("objects"), list):
                raise ValueError("Each alert must contain an objects array.")
            if not all(isinstance(object_id, str) for object_id in alert["objects"]):
                raise ValueError("Alert object IDs must be strings.")


class OutputAlertLayer:
    """Classify, persist, and dispatch validated scene alerts."""

    CHANNELS_BY_LEVEL = {
        "HIGH": ("push", "email", "buzzer"),
        "MEDIUM": ("notification",),
        "LOW": (),
    }
    LEVEL_BY_SEVERITY = {
        "critical": "HIGH",
        "high": "HIGH",
        "warning": "MEDIUM",
        "medium": "MEDIUM",
        "info": "LOW",
        "low": "LOW",
    }
    VALID_HOOKS = {"push", "email", "buzzer", "notification"}

    def __init__(
        self,
        hooks=None,
        database_path=None,
        frame_directory=None,
    ):
        self.hooks = dict(
            getattr(config, "ALERT_HOOKS", {}) if hooks is None else hooks
        )
        invalid_hooks = set(self.hooks) - self.VALID_HOOKS
        if invalid_hooks:
            raise ValueError(
                f"Unsupported alert hook(s): {', '.join(sorted(invalid_hooks))}"
            )
        for channel, callback in self.hooks.items():
            if not callable(callback):
                raise TypeError(f"Alert hook '{channel}' must be callable.")

        self.database_path = Path(
            database_path
            if database_path is not None
            else getattr(config, "ALERT_DATABASE_PATH", "alerts.sqlite3")
        )
        self.frame_directory = Path(
            frame_directory
            if frame_directory is not None
            else getattr(config, "ALERT_FRAME_DIRECTORY", "alert_frames")
        )
        self.database = Database(self.database_path)

    @classmethod
    def classify_severity(cls, severity):
        """Map reasoning severities and explicit level names to alert levels."""
        if not isinstance(severity, str):
            raise ValueError("Alert severity must be a string.")
        level = cls.LEVEL_BY_SEVERITY.get(severity.lower())
        if level is None:
            raise ValueError(f"Unsupported alert severity: {severity!r}")
        return level

    def emit(
        self,
        result,
        frame=None,
        coordinates=None,
        frame_number=None,
        camera_id=None,
    ):
        print("\n[SCENE SUMMARY]")
        print(result["summary"])

        self.emit_alerts(
            result["alerts"],
            frame=frame,
            coordinates=coordinates,
            frame_number=frame_number,
            camera_id=camera_id,
        )

    def emit_alerts(
        self,
        alerts,
        frame=None,
        coordinates=None,
        frame_number=None,
        camera_id=None,
    ):
        if not alerts:
            return

        coordinates = coordinates or {}
        for source_alert in alerts:
            alert = self._normalize_alert(source_alert)
            severity = alert["severity"]
            level = self.classify_severity(severity)
            object_ids = ", ".join(alert["objects"]) or "none"
            related_coordinates = {
                object_id: coordinates[object_id]
                for object_id in alert["objects"]
                if object_id in coordinates
            }
            snapshot_path = None
            if level == "HIGH":
                if frame is None:
                    raise ValueError(
                        "A frame is required to process a HIGH alert."
                    )
                snapshot_path = self._save_frame(frame, frame_number)

            context = {
                "level": level,
                "frame_number": frame_number,
                "camera_id": camera_id,
                "coordinates": related_coordinates,
                "snapshot_path": snapshot_path,
            }
            self._log_alert(alert, context)

            print(
                f"[{level} / {severity.upper()}] "
                f"{alert['title']}: {alert['message']}"
            )
            print(f"  Related objects: {object_ids}")
            if level == "HIGH":
                print(f"  Snapshot: {snapshot_path}")
                print(
                    "  Coordinates: "
                    f"{json.dumps(related_coordinates, sort_keys=True)}"
                )

            for channel in self.CHANNELS_BY_LEVEL[level]:
                callback = self.hooks.get(channel)
                if callback is None:
                    print(
                        f"[ALERT ACTION UNCONFIGURED] {level} alert "
                        f"requires '{channel}' hook."
                    )
                    continue
                try:
                    callback(alert, context)
                except (
                    OSError,
                    TimeoutError,
                    RuntimeError,
                    ValueError,
                    ImportError,
                    smtplib.SMTPException,
                ) as error:
                    print(
                        f"[ALERT ACTION ERROR] {channel} delivery failed: "
                        f"{error}"
                    )

    @classmethod
    def _normalize_alert(cls, alert):
        if "severity" in alert:
            return alert
        level_to_severity = {
            "HIGH": "critical",
            "MEDIUM": "warning",
            "LOW": "info",
        }
        level = alert.get("alert_level")
        if level not in level_to_severity:
            raise ValueError(f"Unsupported alert level: {level!r}")
        return {
            "severity": level_to_severity[level],
            "title": alert.get("alert_type", "CCTV alert"),
            "message": alert.get("reason", ""),
            "objects": alert.get("affected_object_ids", []),
        }

    def _save_frame(self, frame, frame_number):
        self.frame_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        frame_label = (
            f"_frame{frame_number}" if frame_number is not None else ""
        )
        image_path = self.frame_directory / f"alert_{timestamp}{frame_label}.jpg"
        if not cv2.imwrite(str(image_path), frame):
            raise OSError(f"Could not save HIGH-alert frame to {image_path}.")
        return str(image_path)

    def _log_alert(self, alert, context):
        self.database.insert_alert(alert, context)
