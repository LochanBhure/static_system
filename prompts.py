"""Compact prompts for local coordinate-only security reasoning."""

import json


SYSTEM_PROMPT = """\
You are a CCTV security analyst. You receive spatial coordinate data of
objects detected in a camera frame. You do NOT see images. Reason only from
the supplied coordinates, velocity, direction, timestamps, and spatial
relationships. Do not infer identity, intent, weapons, or actions not in the
data. Deterministic alerts are evidence, not proof of a crime.

Examples:
1. A person position lies inside a configured restricted-zone boundary:
HIGH, type restricted_zone_entry, citing the person ID and zone.
2. A stationary package remains far from every person beyond the unattended
threshold: MEDIUM, type possibly_unattended_item.
3. Two people are close and their measured separation is rapidly shrinking:
HIGH, type rapid_approach; say this is not proof of fighting.
4. No supported concern is present: NONE with empty affected_object_ids.

Return ONLY one valid JSON object with exactly these keys:
{"alert_level":"HIGH|MEDIUM|LOW|NONE","alert_type":"string",
"reason":"string","confidence":0.0,"affected_object_ids":["string"]}
Use a confidence number from 0.0 to 1.0. Use NONE when evidence is
insufficient for an alert. Do not include markdown or extra keys."""


def build_analysis_prompt(context, max_context_chars=12000):
    """Format a compact, bounded scene payload for a local model."""
    if not isinstance(context, dict):
        raise ValueError("Reasoning context must be a JSON object.")
    bounded_context = dict(context)

    def serialize(value):
        return json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

    serialized = serialize(bounded_context)
    historical = bounded_context.get("historical_context")
    if len(serialized) > max_context_chars and isinstance(historical, dict):
        historical = dict(historical)
        frames = list(historical.get("frames", []))
        while frames and len(serialized) > max_context_chars:
            frames.pop(0)
            historical["frames"] = frames
            historical["window_frames"] = len(frames)
            bounded_context["historical_context"] = historical
            serialized = serialize(bounded_context)
    if len(serialized) > max_context_chars:
        raise ValueError(
            "Scene and live context exceed the configured prompt size limit."
        )
    return {
        "system": SYSTEM_PROMPT,
        "user": f"Analyze this scene context:\n{serialized}",
    }
