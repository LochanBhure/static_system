"""Server-side multi-camera CCTV application entry point."""

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import signal
import threading
import time
from urllib.error import URLError

import config
from camera_manager import CameraManager
from context_builder import ContextBuilder
from reasoning_engine import ReasoningEngine
from reasoning_pipeline import OutputAlertLayer
from shift_report import ShiftReportGenerator
from providers.buzzer_provider import send_buzzer_alert
from providers.email_provider import send_email_alert
from providers.push_provider import send_push_alert


def _configured_alert_hooks():
    hooks = dict(config.ALERT_HOOKS)
    push_ready = (
        config.PUSH_PROVIDER.lower() == "ntfy" and bool(config.NTFY_TOPIC)
    ) or (
        config.PUSH_PROVIDER.lower() == "pushover"
        and bool(config.PUSHOVER_TOKEN and config.PUSHOVER_USER_KEY)
    ) or bool(config.PUSHOVER_TOKEN and config.PUSHOVER_USER_KEY)
    if push_ready:
        hooks.setdefault("push", send_push_alert)
        hooks.setdefault("notification", send_push_alert)

    if all((
        config.SMTP_HOST,
        config.SMTP_USER,
        config.SMTP_PASS,
        config.ALERT_EMAIL_TO,
    )):
        hooks.setdefault("email", send_email_alert)
    if config.BUZZER_MODE.lower() != "none":
        hooks.setdefault("buzzer", send_buzzer_alert)
    return hooks


def _build_context(builder, snapshot):
    static_objects = snapshot["static_objects"]
    live_objects = snapshot["moving_objects"]
    history = []
    for camera_id, camera_frames in snapshot["history_by_camera"].items():
        for source_frame in camera_frames:
            frame = dict(source_frame)
            moving = frame.get(
                "moving_objects",
                frame.get("live_objects", frame.get("people", [])),
            )
            tagged_moving = []
            for source_object in moving:
                tagged = dict(source_object)
                tagged.setdefault("class", "person")
                tagged["source_object_id"] = tagged.get("id")
                tagged["id"] = (
                    f"{camera_id}:{tagged.get('id', 'unknown')}"
                )
                tagged["camera_id"] = camera_id
                tagged_moving.append(tagged)
            frame["moving_objects"] = tagged_moving
            frame["camera_id"] = camera_id
            history.append(frame)
    history.sort(key=lambda frame: frame.get("timestamp", ""))

    locations = ", ".join(
        f"{camera['name']} ({camera['location']})"
        for camera in snapshot["cameras"]
    )
    scene = {
        "static_objects": static_objects,
        "restricted_zones": snapshot["restricted_zones"],
    }
    context = builder.build_context(
        scene,
        live_objects,
        history[-config.HISTORY_WINDOW_FRAMES * max(1, len(snapshot["cameras"])):],
        camera_id="multi_camera",
        camera_name="Multi-camera system",
        camera_location=locations or "Unknown",
        restricted_zones=snapshot["restricted_zones"],
        timestamp=snapshot["timestamp"],
    )
    context["live_context"]["frame"] = max(
        (frame.get("frame", 0) or 0 for frame in history),
        default=0,
    )
    context["live_context"]["fps"] = 30.0
    context["historical_context"]["cross_camera_events"] = (
        snapshot["cross_camera_events"]
    )
    return context


def run():
    """Start cameras, reasoning, alert delivery, and the shift-report timer."""
    shutdown_event = threading.Event()
    manager = CameraManager()
    context_builder = ContextBuilder()
    reasoning_engine = ReasoningEngine()
    alert_layer = OutputAlertLayer(hooks=_configured_alert_hooks())
    report_generator = ShiftReportGenerator(database=alert_layer.database)
    reasoning_interval = max(
        1.0,
        config.LLM_ANALYSIS_INTERVAL_FRAMES / 30.0,
    )
    report_interval = config.SHIFT_REPORT_INTERVAL_MINUTES * 60.0
    last_llm_time = 0.0
    last_report_time = time.monotonic()
    report_period_start = datetime.now().astimezone()
    executor = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="cctv-background",
    )
    llm_future = None
    llm_job = None
    report_future = None

    def handle_shutdown(signum, frame):
        del signum, frame
        shutdown_event.set()

    previous_sigint = signal.signal(signal.SIGINT, handle_shutdown)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_shutdown)

    def emit_alerts(alerts, coordinate_map, frame_contexts=None):
        for alert in alerts:
            object_ids = alert["affected_object_ids"]
            related = {
                object_id: coordinate_map[object_id]
                for object_id in object_ids
                if object_id in coordinate_map
            }
            camera_id = next(
                (
                    obj["camera_id"]
                    for obj in related.values()
                    if obj.get("camera_id") in manager.pipelines
                ),
                manager.cameras[0]["camera_id"],
            )
            if frame_contexts is None:
                frame, frame_number = manager.latest_frame_context(camera_id)
            else:
                frame, frame_number = frame_contexts[camera_id]
            if frame is None and alert["alert_level"] == "HIGH":
                raise RuntimeError(
                    f"Cannot capture HIGH alert: camera {camera_id} has no frame."
                )
            output_alert = {
                **alert,
                "title": alert["alert_type"].replace("_", " ").title(),
                "message": alert["reason"],
                "objects": object_ids,
            }
            alert_layer.emit_alerts(
                [output_alert],
                frame=frame,
                coordinates=related,
                frame_number=frame_number,
                camera_id=camera_id,
            )

    def collect_frame_contexts():
        return {
            camera["camera_id"]: manager.latest_frame_context(
                camera["camera_id"]
            )
            for camera in manager.cameras
        }

    def process_snapshot(snapshot):
        nonlocal last_llm_time, last_report_time, report_period_start
        nonlocal llm_future, llm_job, report_future
        if shutdown_event.is_set():
            return
        context = _build_context(context_builder, snapshot)
        coordinate_map = {
            obj["id"]: obj
            for obj in (
                snapshot["static_objects"] + snapshot["moving_objects"]
            )
        }
        rule_alerts = reasoning_engine.rules.evaluate(context)
        if rule_alerts:
            emit_alerts(rule_alerts, coordinate_map)

        now_monotonic = time.monotonic()
        llm_due = (
            config.LLM_ENABLED
            and now_monotonic - last_llm_time >= reasoning_interval
        )
        if llm_future is not None and llm_future.done():
            try:
                llm_result = llm_future.result()
            except (URLError, TimeoutError, ValueError) as error:
                print(f"[LLM ERROR] Local reasoning request failed: {error}")
            else:
                prior_rules = llm_job["rule_alerts"]
                combined = reasoning_engine.merge_llm_result(
                    prior_rules,
                    llm_result,
                )
                supplemental = [
                    alert for alert in combined if alert not in prior_rules
                ]
                emit_alerts(
                    supplemental,
                    llm_job["coordinates"],
                    llm_job["frames"],
                )
            llm_future = None
            llm_job = None

        llm_needed = reasoning_engine.should_consult_llm(rule_alerts)
        if llm_due and llm_needed and llm_future is None:
            last_llm_time = now_monotonic
            llm_job = {
                "rule_alerts": list(rule_alerts),
                "coordinates": coordinate_map,
                "frames": collect_frame_contexts(),
            }
            llm_future = executor.submit(
                reasoning_engine.llm_client.analyze,
                context,
            )
        if report_future is not None and report_future.done():
            try:
                report = report_future.result()
            except (URLError, TimeoutError, ValueError) as error:
                print(f"[REPORT ERROR] Could not generate shift report: {error}")
            else:
                print(
                    f"[SHIFT REPORT] {report['total_events']} events from "
                    f"{report['period_start']} to {report['period_end']}"
                )
                report_period_start = datetime.fromisoformat(
                    report["period_end"]
                )
            report_future = None

        if (
            now_monotonic - last_report_time >= report_interval
            and report_future is None
        ):
            report_end = datetime.now().astimezone()
            report_future = executor.submit(
                report_generator.generate_report,
                report_period_start,
                report_end,
            )
            last_report_time = now_monotonic

    try:
        print(
            f"[START] CCTV system ready for "
            f"{len(manager.cameras)} camera(s). Press Ctrl+C to stop."
        )
        manager.run(on_snapshot=process_snapshot, stop_event=shutdown_event)
    finally:
        shutdown_event.set()
        manager.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    run()
