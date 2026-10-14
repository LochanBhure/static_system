"""Deterministic CCTV rules with local Ollama reasoning as a fallback."""

import json
from collections import deque
from datetime import datetime, time, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

import config
from prompts import build_analysis_prompt


class RuleBasedReasoningEngine:
    """Evaluate configured spatial and temporal conditions without an LLM."""

    def __init__(self):
        self.active_conditions = set()
        self.loiter_state = {}

    def evaluate(self, context, *, now=None):
        scene = context.get("scene_context", {})
        live = context.get("live_context", {})
        historical = context.get("historical_context", {})
        moving = live.get("moving_objects", [])
        static = scene.get("static_objects", [])
        zones = scene.get("restricted_zones", [])
        frame = live.get("frame")
        fps = float(live.get("fps", 30.0) or 30.0)
        current_time = now or datetime.now().astimezone()
        alerts = []
        conditions = set()

        people = [obj for obj in moving if obj.get("class") == "person"]
        for person in people:
            position = self._position(person)
            for zone in zones:
                zone_camera = zone.get("camera_id")
                if (
                    zone_camera is not None
                    and person.get("camera_id") != zone_camera
                ):
                    continue
                bounds = zone.get("bbox", zone.get("bounds"))
                if bounds and self._inside(position, bounds):
                    name = zone.get("name", "restricted zone")
                    condition = ("restricted_zone", person.get("id"), name)
                    conditions.add(condition)
                    alerts.append(self._alert(
                        "HIGH",
                        "restricted_zone_entry",
                        f"Person entered restricted zone {name}.",
                        [person.get("id")],
                        0.95,
                    ))

        for obj in static:
            if obj.get("class") not in config.ABANDONED_ITEM_CLASSES:
                continue
            elapsed = self._seconds_since_seen(obj, frame, fps)
            position = self._position(obj)
            attended = any(
                (
                    obj.get("camera_id") is None
                    or person.get("camera_id") == obj.get("camera_id")
                )
                and
                self._distance(position, self._position(person))
                <= config.PERSON_ITEM_DISTANCE_NORMALIZED
                for person in people
            )
            if elapsed >= config.ABANDONED_OBJECT_SECONDS and not attended:
                condition = ("abandoned_item", obj.get("id"))
                conditions.add(condition)
                alerts.append(self._alert(
                    "MEDIUM",
                    "possibly_unattended_item",
                    f"{obj.get('class')} has been unattended for "
                    f"{elapsed:.1f} seconds.",
                    [obj.get("id")],
                    0.85,
                ))

        for pair in self._rapid_approaches(people, historical, frame, fps):
            conditions.add(("rapid_approach", *pair))
            alerts.append(self._alert(
                "HIGH",
                "rapid_approach",
                "Two people are rapidly converging; this indicates motion "
                "and proximity, not confirmed fighting.",
                list(pair),
                0.8,
            ))

        self.loiter_state = {
            object_id: state
            for object_id, state in self.loiter_state.items()
            if (
                current_time - state["last_seen"]
            ).total_seconds() <= config.LOITER_TRACK_TIMEOUT_SECONDS
        }
        for person in people:
            object_id = person.get("id")
            if self._update_loitering(person, current_time):
                condition = ("loitering", object_id)
                conditions.add(condition)
                alerts.append(self._alert(
                    "MEDIUM",
                    "possible_loitering",
                    "Person remained within a small area for an extended "
                    "period.",
                    [object_id],
                    0.7,
                ))

        for obj in moving:
            velocity = self._velocity(obj)
            if velocity > config.VELOCITY_THRESHOLD:
                object_id = obj.get("id")
                conditions.add(("high_velocity", object_id))
                alerts.append(self._alert(
                    "MEDIUM",
                    "high_velocity",
                    f"{obj.get('class', 'Object')} velocity {velocity:.2f} "
                    f"exceeds the configured threshold "
                    f"{config.VELOCITY_THRESHOLD:.2f}.",
                    [object_id],
                    0.8,
                ))

        crowded_people = self._crowded_people(people)
        if len(crowded_people) >= config.CROWDING_MIN_PEOPLE:
            condition = ("crowding", *sorted(crowded_people))
            conditions.add(condition)
            alerts.append(self._alert(
                "LOW",
                "crowding",
                f"{len(crowded_people)} people are clustered in the scene.",
                sorted(crowded_people),
                0.65,
            ))

        if moving and self._is_after_hours(current_time):
            condition = ("after_hours",)
            conditions.add(condition)
            alerts.append(self._alert(
                "HIGH",
                "after_hours_activity",
                "Moving objects were detected during configured after-hours.",
                [obj.get("id") for obj in moving],
                0.9,
            ))

        new_alerts = []
        for alert in alerts:
            condition = self._condition_for(alert)
            if condition not in self.active_conditions:
                new_alerts.append(alert)
            conditions.add(condition)
        self.active_conditions = conditions
        return new_alerts

    @staticmethod
    def _alert(level, alert_type, reason, object_ids, confidence):
        return {
            "alert_level": level,
            "alert_type": alert_type,
            "reason": reason,
            "confidence": confidence,
            "affected_object_ids": [
                object_id for object_id in object_ids if isinstance(object_id, str)
            ],
        }

    @staticmethod
    def _condition_for(alert):
        return (
            alert["alert_type"],
            *sorted(alert["affected_object_ids"]),
        )

    @staticmethod
    def _position(obj):
        position = obj.get("position", obj.get("coordinates", obj))
        return {
            "x": position.get("nx", position.get("x", 0.0)),
            "y": position.get("ny", position.get("y", 0.0)),
        }

    @staticmethod
    def _pixel_position(obj):
        position = obj.get("position", obj.get("coordinates", obj))
        return {
            "x": position.get("x", position.get("nx", 0.0)),
            "y": position.get("y", position.get("ny", 0.0)),
        }

    @staticmethod
    def _inside(position, bounds):
        if isinstance(bounds, dict):
            left = bounds.get("x1", bounds.get("left", 0.0))
            top = bounds.get("y1", bounds.get("top", 0.0))
            right = bounds.get("x2", bounds.get("right", 0.0))
            bottom = bounds.get("y2", bounds.get("bottom", 0.0))
        else:
            left, top, right, bottom = bounds
        return left <= position["x"] <= right and top <= position["y"] <= bottom

    @staticmethod
    def _distance(first, second):
        return float(np.hypot(first["x"] - second["x"], first["y"] - second["y"]))

    @staticmethod
    def _seconds_since_seen(obj, frame, fps):
        scene_info = obj.get("scene_info", {})
        if isinstance(obj.get("first_seen_seconds_ago"), (int, float)):
            return float(obj["first_seen_seconds_ago"])
        frame = obj.get("camera_frame", frame)
        fps = float(obj.get("camera_fps", fps) or fps)
        first_frame = scene_info.get(
            "first_seen_frame", obj.get("first_seen_frame", frame or 0)
        )
        return max(0.0, (float(frame or 0) - float(first_frame)) / fps)

    @staticmethod
    def _velocity(obj):
        scalar = obj.get("speed_body_lengths_per_second")
        if isinstance(scalar, (int, float)):
            return float(scalar)
        velocity = obj.get("velocity", 0.0)
        if isinstance(velocity, dict):
            return float(np.hypot(
                float(velocity.get("vx", 0.0)),
                float(velocity.get("vy", 0.0)),
            ))
        return float(velocity or 0.0)

    def _update_loitering(self, person, now):
        object_id = person.get("id")
        position = self._pixel_position(person)
        state = self.loiter_state.get(object_id)
        if state is None:
            state = {"positions": deque(), "last_seen": now}
            self.loiter_state[object_id] = state
        state["last_seen"] = now
        state["positions"].append((now, position))
        cutoff = now - timedelta(seconds=config.LOITER_TIME_SECONDS)
        while state["positions"] and state["positions"][0][0] < cutoff:
            state["positions"].popleft()

        observations = state["positions"]
        if len(observations) < config.LOITER_MIN_OBSERVATIONS:
            return False
        elapsed = (observations[-1][0] - observations[0][0]).total_seconds()
        center = {
            "x": sum(item[1]["x"] for item in observations) / len(observations),
            "y": sum(item[1]["y"] for item in observations) / len(observations),
        }
        remains_in_area = all(
            self._distance(item[1], center) <= config.LOITER_RADIUS
            for item in observations
        )
        return elapsed >= config.LOITER_TIME_SECONDS and remains_in_area

    @classmethod
    def _rapid_approaches(cls, people, historical, frame, fps):
        prior_by_id = {}
        for history_frame in historical.get("frames", []):
            for obj in history_frame.get(
                "moving_objects",
                history_frame.get("live_objects", history_frame.get("people", [])),
            ):
                if obj.get("class") == "person":
                    prior_by_id[obj["id"]] = (
                        cls._pixel_position(obj),
                        history_frame.get("frame"),
                        float(obj.get("body_height_px", 1.0) or 1.0),
                    )
        detected_pairs = []
        for index, first in enumerate(people):
            for second in people[index + 1:]:
                if first.get("camera_id") != second.get("camera_id"):
                    continue
                first_prior = prior_by_id.get(first.get("id"))
                second_prior = prior_by_id.get(second.get("id"))
                if first_prior is None or second_prior is None:
                    continue
                current_frame = min(
                    float(first.get("camera_frame", frame or 0)),
                    float(second.get("camera_frame", frame or 0)),
                )
                prior_frame = max(
                    value[1] for value in (first_prior, second_prior)
                    if isinstance(value[1], (int, float))
                ) if all(
                    isinstance(value[1], (int, float))
                    for value in (first_prior, second_prior)
                ) else None
                if prior_frame is None or current_frame <= prior_frame:
                    continue
                first_now = cls._pixel_position(first)
                second_now = cls._pixel_position(second)
                first_before, second_before = first_prior[0], second_prior[0]
                current_mean_height = max(
                    (
                        float(first.get("body_height_px", 1.0) or 1.0)
                        + float(second.get("body_height_px", 1.0) or 1.0)
                    ) / 2,
                    1.0,
                )
                previous_mean_height = max(
                    (first_prior[2] + second_prior[2]) / 2,
                    1.0,
                )
                current_distance = (
                    cls._distance(first_now, second_now) / current_mean_height
                )
                previous_distance = (
                    cls._distance(first_before, second_before)
                    / previous_mean_height
                )
                camera_fps = float(first.get("camera_fps", fps) or fps)
                elapsed = (current_frame - prior_frame) / camera_fps
                closing_speed = (previous_distance - current_distance) / elapsed
                if (
                    current_distance <= config.RAPID_APPROACH_DISTANCE_BODY_LENGTHS
                    and closing_speed >= (
                        config.RAPID_APPROACH_SPEED_BODY_LENGTHS_PER_SECOND
                    )
                ):
                    detected_pairs.append(tuple(sorted((
                        first["id"], second["id"]
                    ))))
        return detected_pairs

    @classmethod
    def _crowded_people(cls, people):
        near = set()
        for index, first in enumerate(people):
            for second in people[index + 1:]:
                if first.get("camera_id") != second.get("camera_id"):
                    continue
                if cls._distance(
                    cls._position(first), cls._position(second)
                ) <= config.CROWDING_RADIUS_NORMALIZED:
                    near.update((first["id"], second["id"]))
        return near

    @staticmethod
    def _is_after_hours(now):
        start = time.fromisoformat(config.AFTER_HOURS_START)
        end = time.fromisoformat(config.AFTER_HOURS_END)
        current = now.timetz().replace(tzinfo=None)
        if start <= end:
            return start <= current < end
        return current >= start or current < end


class OllamaReasoningClient:
    """Request a schema-validated result from a local Ollama chat endpoint."""

    def __init__(self, host=None, model=None, timeout=None):
        self.endpoint = (
            host or config.OLLAMA_HOST
        ).rstrip("/") + "/api/chat"
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT

    def analyze(self, context):
        prompt = build_analysis_prompt(context)
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        if not isinstance(body, dict):
            raise ValueError("Ollama response must be a JSON object.")
        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), str
        ):
            raise ValueError("Ollama response is missing message content.")
        content = message["content"]
        result = json.loads(content)
        self.validate_result(result)
        known_ids = {
            obj["id"]
            for collection in (
                context.get("scene_context", {}).get("static_objects", []),
                context.get("live_context", {}).get("moving_objects", []),
            )
            for obj in collection
            if isinstance(obj.get("id"), str)
        }
        unknown_ids = set(result["affected_object_ids"]) - known_ids
        if unknown_ids:
            raise ValueError(
                "Ollama response references unknown object IDs: "
                + ", ".join(sorted(unknown_ids))
            )
        return result

    @staticmethod
    def validate_result(result):
        if not isinstance(result, dict):
            raise ValueError("Reasoning result must be a JSON object.")
        expected_fields = {
            "alert_level",
            "alert_type",
            "reason",
            "confidence",
            "affected_object_ids",
        }
        if set(result) != expected_fields:
            raise ValueError(
                "Reasoning result must contain exactly the documented fields."
            )
        if result.get("alert_level") not in {"HIGH", "MEDIUM", "LOW", "NONE"}:
            raise ValueError("alert_level must be HIGH, MEDIUM, LOW, or NONE.")
        if not isinstance(result.get("alert_type"), str):
            raise ValueError("alert_type must be a string.")
        if not isinstance(result.get("reason"), str):
            raise ValueError("reason must be a string.")
        confidence = result.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("confidence must be a number from 0.0 to 1.0.")
        object_ids = result.get("affected_object_ids")
        if not isinstance(object_ids, list) or not all(
            isinstance(object_id, str) for object_id in object_ids
        ):
            raise ValueError("affected_object_ids must be an array of strings.")


class ReasoningEngine:
    """Run deterministic rules first and consult Ollama when no rule fires."""

    def __init__(self, rules=None, llm_client=None):
        self.rules = rules or RuleBasedReasoningEngine()
        self.llm_client = llm_client or OllamaReasoningClient()

    @staticmethod
    def should_consult_llm(rule_alerts):
        return not rule_alerts or any(
            alert["confidence"] < config.LLM_UNCERTAIN_CONFIDENCE_THRESHOLD
            for alert in rule_alerts
        )

    @staticmethod
    def merge_llm_result(rule_alerts, llm_result):
        if llm_result is None or llm_result["alert_level"] == "NONE":
            return list(rule_alerts)
        if any(
            alert["alert_type"] == llm_result["alert_type"]
            and alert["affected_object_ids"]
            == llm_result["affected_object_ids"]
            for alert in rule_alerts
        ):
            return list(rule_alerts)
        return [*rule_alerts, llm_result]

    def analyze(self, context, *, now=None, include_llm=True):
        rule_alerts = self.rules.evaluate(context, now=now)
        if not self.should_consult_llm(rule_alerts):
            return rule_alerts
        if not include_llm or not config.LLM_ENABLED:
            return rule_alerts
        try:
            llm_result = self.llm_client.analyze(context)
        except (URLError, TimeoutError, ValueError) as error:
            print(f"[LLM ERROR] Local reasoning request failed: {error}")
            return rule_alerts
        return self.merge_llm_result(rule_alerts, llm_result)
