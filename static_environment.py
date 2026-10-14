# ============================================
# STATIC ENVIRONMENT SYSTEM
# ============================================
# PURPOSE  : Detect only NON-MOVING objects
# OUTPUT   : Coordinate vectors (JSON + Text)
#            Ready to feed into Text LLM
# CONCEPT  : Reverse Game Engine Approach
#            Image → Coordinates (like 3D space)
# ============================================

import cv2
import numpy as np
import json
from collections import deque
from datetime import datetime
from urllib.error import URLError
from ultralytics import YOLO
import config
from reasoning_pipeline import (
    ContextLayer,
    LLMReasoningLayer,
    OutputAlertLayer,
    RuleBasedReasoningLayer,
)


# ============================================================
# STATIC ENVIRONMENT SYSTEM CLASS
# ============================================================

class StaticEnvironmentSystem:

    def __init__(
        self,
        camera_config=None,
        enable_reasoning=True,
        display=True,
    ):
        self.camera_config = camera_config or {}
        self.camera_id = self.camera_config.get("camera_id", "camera_1")
        self.camera_name = self.camera_config.get("name", "Camera 1")
        self.camera_location = self.camera_config.get(
            "location", config.CAMERA_LOCATION
        )
        self.restricted_zones = self.camera_config.get(
            "restricted_zones", config.RESTRICTED_ZONES
        )
        self.enable_reasoning = enable_reasoning
        self.display = display
        print("=" * 50)
        print("  STATIC ENVIRONMENT SYSTEM STARTING...")
        print("=" * 50)

        # --- Camera Setup ---
        camera_source = self.camera_config.get(
            "source", self.camera_config.get("url", config.CAMERA_SOURCE)
        )
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                f"[ERROR] Cannot open camera source: {camera_source!r}"
            )
        print(f"[✅] Camera Connected: {camera_source}")

        # --- YOLO Model (Lightweight Nano) ---
        try:
            self.model = YOLO(config.MODEL_PATH)
        except Exception:
            self.cap.release()
            raise
        print(f"[✅] YOLO Model Loaded: {config.MODEL_PATH}")

        # --- Background Subtractor ---
        # This is the CORE of static detection
        # It learns what the background looks like
        # Anything different = motion
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history         = config.BG_HISTORY,
            varThreshold    = config.BG_THRESHOLD,
            detectShadows   = config.BG_DETECT_SHADOWS
        )
        print(f"[✅] Background Subtractor Ready")

        # --- Static Object Cache ---
        # This is the MAIN CACHE
        # Once object is confirmed static = stored here
        # Only updates when something CHANGES in scene
        self.static_objects     = {}   # Final confirmed static objects
        self.candidate_tracker  = {}   # Objects being tracked for confirmation

        self.context_layer = ContextLayer()
        self.reasoning_layer = LLMReasoningLayer(
            config.OLLAMA_BASE_URL,
            config.OLLAMA_MODEL,
            config.OLLAMA_TIMEOUT_SECONDS,
        )
        self.output_layer = OutputAlertLayer() if enable_reasoning else None
        self.rule_layer = RuleBasedReasoningLayer()
        self.current_rule_alerts = []

        # --- Frame Counter ---
        self.frame_count  = 0
        self.cache_version = 0
        self.latest_frame = None
        self.live_tracks = {}
        self.current_live_objects = []
        self.next_live_track_id = 1
        self.next_person_id = 1
        self.frame_history = deque(maxlen=config.HISTORY_WINDOW_FRAMES)
        self.recent_behavior_events = deque(
            maxlen=config.HISTORY_WINDOW_FRAMES
        )
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0
        self.last_reasoning_frame = 0

        print("[✅] Static Environment System Ready!\n")


    # ============================================================
    # CORE FUNCTION 1 - GET MOTION MASK
    # Tells us WHERE motion is happening in frame
    # ============================================================

    def get_motion_mask(self, frame):
        """
        Returns binary mask:
        WHITE = motion detected area
        BLACK = no motion (static area)
        """

        # Apply background subtractor
        fg_mask = self.bg_subtractor.apply(frame)

        # Clean up noise with morphology operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Remove small noise dots
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Fill small gaps
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        return fg_mask


    # ============================================================
    # CORE FUNCTION 2 - CHECK IF OBJECT IS MOVING
    # Compares object bbox against motion mask
    # ============================================================

    def is_object_moving(self, bbox, motion_mask):
        """
        Checks if a detected object overlaps with moving areas

        Returns:
            True  = Object is MOVING  → IGNORE in static system
            False = Object is STATIC  → KEEP in static system
        """

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Crop motion mask to object region only
        frame_height, frame_width = motion_mask.shape[:2]
        x1 = max(0, min(x1, frame_width))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height))
        y2 = max(0, min(y2, frame_height))
        object_region = motion_mask[y1:y2, x1:x2]

        if object_region.size == 0:
            return False

        # Calculate what % of object area has motion
        motion_pixels = np.count_nonzero(object_region == 255)
        total_pixels  = object_region.size
        motion_ratio  = motion_pixels / total_pixels

        # If motion > threshold = object is moving
        return motion_ratio > config.MOTION_THRESHOLD


    # ============================================================
    # CORE FUNCTION 3 - ESTIMATE DEPTH (Z Coordinate)
    # Uses perspective geometry like a game engine
    # Objects lower in frame = closer to camera
    # ============================================================

    def estimate_depth(self, y_bottom, frame_height):
        """
        Estimates Z depth from vertical position

        Game Engine Logic:
        - Objects at bottom of frame = CLOSE  = Low Z value
        - Objects at top of frame   = FAR     = High Z value

        Returns: Z value (0-100)
        """
        depth = (1 - (y_bottom / frame_height)) * 100
        return round(depth, 2)


    # ============================================================
    # CORE FUNCTION 4 - EXTRACT COORDINATE VECTOR
    # THE REVERSE GAME ENGINE CONVERSION
    # Image Object → 3D Coordinate Vector
    # ============================================================

    def extract_coordinates(self, detection, frame_shape, obj_id):
        """
        Converts visual detection into coordinate vector

        THIS IS THE CORE CONCEPT:
        Just like game engine stores objects as X,Y,Z
        We convert image objects into X,Y,Z coordinates
        = Computer's NATIVE language
        = Ready for Text LLM to reason on
        """

        frame_height, frame_width = frame_shape[:2]

        x1, y1, x2, y2 = detection['bbox']

        # --- Center Point (X, Y) ---
        cx = round((x1 + x2) / 2, 2)
        cy = round((y1 + y2) / 2, 2)

        # --- Normalized Coordinates (0.0 to 1.0) ---
        # Useful for LLM reasoning (position independent of resolution)
        nx = round(cx / frame_width,  4)
        ny = round(cy / frame_height, 4)

        # --- Z Depth Estimation ---
        z = self.estimate_depth(y2, frame_height)

        # --- Object Size ---
        width  = round(x2 - x1, 2)
        height = round(y2 - y1, 2)

        # --- Scene Zone (helps LLM understand location) ---
        zone = self.get_scene_zone(nx, ny)

        # --- Build Complete Coordinate Vector ---
        return {
            "id"         : obj_id,
            "class"      : detection['class'],
            "confidence" : round(detection['confidence'], 3),
            "status"     : "static",

            "coordinates": {
                "x"  : round(cx, 2),  # Pixel X center
                "y"  : round(cy, 2),  # Pixel Y center
                "z"  : z,             # Estimated depth
                "nx" : nx,            # Normalized X (0-1)
                "ny" : ny             # Normalized Y (0-1)
            },

            "dimensions": {
                "width"  : round(width,  2),
                "height" : round(height, 2),
                "area"   : round(width * height, 2),
                "bbox"   : [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]
            },

            "scene_info": {
                "zone"               : zone,
                "first_seen_frame"   : self.frame_count,
                "timestamp"          : datetime.now().isoformat(),
                "cache_version"      : self.cache_version
            }
        }


    # ============================================================
    # HELPER - GET SCENE ZONE
    # Divides frame into zones for LLM understanding
    # ============================================================

    def get_scene_zone(self, nx, ny):
        """
        Divides scene into 9 zones like a grid:
        ┌──────────┬──────────┬──────────┐
        │ TOP-LEFT │ TOP-MID  │ TOP-RIGHT│
        ├──────────┼──────────┼──────────┤
        │ MID-LEFT │  CENTER  │ MID-RIGHT│
        ├──────────┼──────────┼──────────┤
        │ BOT-LEFT │ BOT-MID  │ BOT-RIGHT│
        └──────────┴──────────┴──────────┘
        """
        if ny < 0.33:
            v = "TOP"
        elif ny < 0.66:
            v = "MIDDLE"
        else:
            v = "BOTTOM"

        if nx < 0.33:
            h = "LEFT"
        elif nx < 0.66:
            h = "CENTER"
        else:
            h = "RIGHT"

        return f"{v}-{h}"


    # ============================================================
    # HELPER - IoU (Intersection over Union)
    # Checks if two bounding boxes are the same object
    # ============================================================

    def calculate_iou(self, box1, box2):
        """
        Calculates overlap between two bounding boxes
        Returns 0.0 (no overlap) to 1.0 (perfect overlap)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


    # ============================================================
    # CORE FUNCTION 5 - UPDATE STATIC CACHE
    # The BRAIN of the static system
    # Only updates when scene ACTUALLY changes
    # ============================================================

    def update_static_cache(self, static_detections, frame_shape):
        """
        Manages the static object cache:

        Logic:
        1. Existing cached objects still there? → Keep them
        2. Cached object disappeared?           → Remove it
        3. New non-moving object seen?          → Track it
        4. Tracked object stable for N frames?  → Add to cache
        """

        cache_changed = False

        # -------------------------------------------
        # STEP 1 - Check if cached objects disappeared
        # -------------------------------------------
        objects_to_remove = []

        for cached_id, cached_obj in self.static_objects.items():
            cached_bbox = cached_obj['dimensions']['bbox']
            still_present = False

            for det in static_detections:
                iou = self.calculate_iou(cached_bbox, det['bbox'])
                if iou > 0.4:
                    still_present = True
                    break

            if not still_present:
                objects_to_remove.append(cached_id)
                cache_changed = True
                print(f"[CHANGE DETECTED] Object LEFT scene: "
                      f"{self.static_objects[cached_id]['class']} (ID: {cached_id})")

        # Remove disappeared objects from cache
        for obj_id in objects_to_remove:
            del self.static_objects[obj_id]


        # -------------------------------------------
        # STEP 2 - Track new potential static objects
        # -------------------------------------------
        for det in static_detections:
            bbox = det['bbox']

            # Check if already in main cache
            in_main_cache = False
            for cached_obj in self.static_objects.values():
                if self.calculate_iou(bbox, cached_obj['dimensions']['bbox']) > 0.4:
                    in_main_cache = True
                    break

            if in_main_cache:
                continue

            # Track as candidate
            # Key = class + position (rough)
            obj_key = f"{det['class']}_{int(bbox[0]/20)}_{int(bbox[1]/20)}"

            if obj_key not in self.candidate_tracker:
                # New candidate spotted
                self.candidate_tracker[obj_key] = {
                    'count'     : 1,
                    'detection' : det,
                    'first_seen': self.frame_count
                }
            else:
                # Increment stable frame count
                self.candidate_tracker[obj_key]['count'] += 1

                # Enough frames to CONFIRM static?
                if self.candidate_tracker[obj_key]['count'] >= config.STATIC_CONFIRM_FRAMES:

                    # YES! Add to static cache
                    obj_id = f"static_{self.cache_version + 1}"
                    coord_vector = self.extract_coordinates(det, frame_shape, obj_id)

                    self.static_objects[obj_id] = coord_vector

                    # Cleanup candidate
                    del self.candidate_tracker[obj_key]

                    self.cache_version += 1
                    cache_changed = True

                    print(f"[NEW STATIC OBJECT] Confirmed: "
                          f"{det['class']} at Zone:{coord_vector['scene_info']['zone']} "
                          f"(ID: {obj_id})")

        # Cleanup candidates that are no longer detected
        candidates_to_remove = []
        for key in self.candidate_tracker:
            candidate_bbox = self.candidate_tracker[key]['detection']['bbox']
            still_detected = False
            for det in static_detections:
                if self.calculate_iou(candidate_bbox, det['bbox']) > 0.4:
                    still_detected = True
                    break
            if not still_detected:
                candidates_to_remove.append(key)

        for key in candidates_to_remove:
            del self.candidate_tracker[key]

        return cache_changed


    # ============================================================
    # OUTPUT FUNCTION 1 - JSON FORMAT
    # Raw structured data
    # ============================================================

    def get_json_output(self):
        """Returns complete static cache as JSON"""
        return json.dumps({
            "system"         : "static_environment",
            "frame"          : self.frame_count,
            "timestamp"      : datetime.now().isoformat(),
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "camera_location": self.camera_location,
            "restricted_zones": self.restricted_zones,
            "total_objects"  : len(self.static_objects),
            "objects"        : self.static_objects
        }, indent=2)


    # ============================================================
    # OUTPUT FUNCTION 2 - TEXT FORMAT FOR LLM
    # Human readable text that LLM can reason on
    # ============================================================

    def get_llm_text_output(self):
        """
        Converts coordinate cache into TEXT for LLM

        This is what gets sent to your Text LLM!
        The LLM reads this like reading a scene description
        but in coordinate/spatial format
        """

        if not self.static_objects:
            return "STATIC ENVIRONMENT: Empty scene - no static objects detected."

        text  = "=" * 40 + "\n"
        text += "STATIC ENVIRONMENT SNAPSHOT\n"
        text += "=" * 40 + "\n"
        text += f"Frame     : {self.frame_count}\n"
        text += f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"Total Static Objects: {len(self.static_objects)}\n"
        text += "-" * 40 + "\n\n"

        for obj_id, obj in self.static_objects.items():
            text += f"[OBJECT] ID: {obj_id}\n"
            text += f"  Type       : {obj['class']}\n"
            text += f"  Confidence : {obj['confidence']}\n"
            text += f"  Position   : X={obj['coordinates']['x']}, "
            text += f"Y={obj['coordinates']['y']}, "
            text += f"Z(depth)={obj['coordinates']['z']}\n"
            text += f"  Norm Pos   : NX={obj['coordinates']['nx']}, "
            text += f"NY={obj['coordinates']['ny']}\n"
            text += f"  Size       : W={obj['dimensions']['width']}px, "
            text += f"H={obj['dimensions']['height']}px\n"
            text += f"  Zone       : {obj['scene_info']['zone']}\n"
            text += f"  Since Frame: {obj['scene_info']['first_seen_frame']}\n"
            text += "\n"

        return text


    # ============================================================
    # VISUALIZATION - Draw on frame
    # ============================================================

    def draw_visualization(self, frame):
        """Draw static objects and info on frame"""

        if config.SHOW_COORDS:
            for obj in self.static_objects.values():
                bbox = obj['dimensions']['bbox']
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)

                label1 = f"[STATIC] {obj['class']}"
                label2 = (
                    f"X:{obj['coordinates']['x']:.0f} "
                    f"Y:{obj['coordinates']['y']:.0f} "
                    f"Z:{obj['coordinates']['z']:.0f}"
                )
                label3 = f"Zone: {obj['scene_info']['zone']}"

                cv2.putText(frame, label1, (x1, y1 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 2)
                cv2.putText(frame, label2, (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 0), 1)
                cv2.putText(frame, label3, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 0), 1)

            for candidate in self.candidate_tracker.values():
                bbox = candidate['detection']['bbox']
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])
                progress = min(
                    candidate['count'] / config.STATIC_CONFIRM_FRAMES, 1.0
                )
                label = f"Confirming... {int(progress * 100)}%"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # Top info bar
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 600), 60), (0, 0, 0), -1)
        cv2.putText(frame, f"STATIC SYSTEM | Frame: {self.frame_count}",
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Confirmed Static: {len(self.static_objects)} | Candidates: {len(self.candidate_tracker)}",
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 100), 1)

        return frame


    # ============================================================
    # MAIN LOOP
    # ============================================================

    def process_frame(self, frame):
        """Detect static and live objects, then update scene and history state."""
        self.latest_frame = frame
        motion_mask = self.get_motion_mask(frame)
        results = self.model(
            frame,
            conf=config.MIN_CONFIDENCE,
            verbose=False,
            device=config.VISION_DEVICE,
        )
        static_detections = []
        live_detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                confidence = float(box.conf[0].item())
                if confidence < config.MIN_CONFIDENCE:
                    continue

                bbox = [float(value) for value in box.xyxy[0].tolist()]
                class_id = int(box.cls[0].item())
                detection = {
                    'bbox': bbox,
                    'class': self.model.names[class_id],
                    'confidence': confidence
                }
                moving = self.is_object_moving(bbox, motion_mask)
                if detection['class'] == "person":
                    live_detections.append(detection)
                elif moving:
                    live_detections.append(detection)
                else:
                    static_detections.append(detection)

        cache_changed = self.update_static_cache(static_detections, frame.shape)
        self.current_live_objects = self.update_live_tracks(live_detections, frame.shape)
        self.frame_history.append({
            "frame": self.frame_count,
            "timestamp": datetime.now().astimezone().isoformat(),
            "camera_id": self.camera_id,
            "fps": self.fps,
            "static_object_count": len(self.static_objects),
            "live_object_count": len(self.current_live_objects),
            "moving_objects": [
                obj.copy() for obj in self.current_live_objects
            ],
            "people": [
                {
                    "id": obj["id"],
                    "position": obj["position"],
                    "status": obj["status"],
                    "direction": obj["direction"],
                    "speed_body_lengths_per_second": (
                        obj["speed_body_lengths_per_second"]
                    ),
                    "locomotion": obj["locomotion"],
                }
                for obj in self.current_live_objects
                if obj["class"] == "person"
            ],
            "behavior_events": [
                event.copy() for event in self.recent_behavior_events
                if event["frame"] == self.frame_count
            ],
            "rule_alerts": [],
        })
        if self.enable_reasoning:
            self.current_rule_alerts = self.rule_layer.evaluate(
                json.loads(self.get_json_output()),
                self.current_live_objects,
                self.frame_count,
                self.fps,
            )
            self.frame_history[-1]["rule_alerts"] = [
                alert.copy() for alert in self.current_rule_alerts
            ]
            self.output_layer.emit_alerts(
                self.current_rule_alerts,
                frame=frame,
                coordinates=self.get_alert_coordinates(self.current_rule_alerts),
                frame_number=self.frame_count,
                camera_id=self.camera_id,
            )
        analysis_due = (
            self.frame_count - self.last_reasoning_frame
            >= config.LLM_ANALYSIS_INTERVAL_FRAMES
        )
        if (
            self.enable_reasoning
            and config.LLM_ENABLED
            and (cache_changed or analysis_due)
        ):
            self.run_reasoning_pipeline()
        return motion_mask

    def get_alert_coordinates(self, alerts):
        """Return current scene coordinates for object IDs referenced by alerts."""
        objects_by_id = {}
        for history_frame in self.frame_history:
            for person in history_frame["people"]:
                objects_by_id[person["id"]] = {
                    "id": person["id"],
                    "position": person["position"],
                    "frame": history_frame["frame"],
                }
        objects_by_id.update(
            {obj["id"]: obj for obj in self.static_objects.values()}
        )
        objects_by_id.update(
            {obj["id"]: obj for obj in self.current_live_objects}
        )
        object_ids = {
            object_id
            for alert in alerts
            for object_id in alert["objects"]
        }
        return {
            object_id: objects_by_id[object_id]
            for object_id in object_ids
            if object_id in objects_by_id
        }

    def update_live_tracks(self, detections, frame_shape):
        """Associate live detections and describe position and movement."""
        frame_height, frame_width = frame_shape[:2]
        diagonal = float(np.hypot(frame_width, frame_height))
        unmatched_tracks = set(self.live_tracks)

        for detection in detections:
            bbox = detection["bbox"]
            center = (
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2,
            )
            best_id = None
            best_distance = diagonal * 0.12

            for track_id in unmatched_tracks:
                track = self.live_tracks[track_id]
                if track["class"] != detection["class"]:
                    continue
                old_position = track["positions"][-1]
                old_center = old_position[:2]
                distance = float(np.hypot(
                    center[0] - old_center[0],
                    center[1] - old_center[1],
                ))
                if distance < best_distance:
                    best_id = track_id
                    best_distance = distance

            if best_id is None:
                if detection["class"] == "person":
                    track_id = f"Person{self.next_person_id}"
                    self.next_person_id += 1
                else:
                    track_id = f"live_{self.next_live_track_id}"
                    self.next_live_track_id += 1
                self.live_tracks[track_id] = {
                    "class": detection["class"],
                    "confidence": detection["confidence"],
                    "bbox": bbox,
                    "first_seen_frame": self.frame_count,
                    "last_seen_frame": self.frame_count,
                    "positions": deque(
                        maxlen=config.HISTORY_WINDOW_FRAMES + 1
                    ),
                    "speed_body_lengths_per_second": 0.0,
                    "velocity_vector": {"vx": 0.0, "vy": 0.0, "vz": 0.0},
                    "previous_speed_body_lengths_per_second": 0.0,
                    "locomotion": "observed",
                }
                best_id = track_id
            else:
                unmatched_tracks.remove(best_id)
                track = self.live_tracks[best_id]
                track["confidence"] = detection["confidence"]
                track["bbox"] = bbox
                track["last_seen_frame"] = self.frame_count

            track = self.live_tracks[best_id]
            previous_positions = list(track["positions"])
            if previous_positions:
                previous_position = previous_positions[-1]
                previous_center = previous_position[:2]
                previous_height = previous_position[2]
                previous_frame = previous_position[3]
                elapsed_seconds = (self.frame_count - previous_frame) / self.fps
                average_height = max((previous_height + (bbox[3] - bbox[1])) / 2, 1.0)
                if elapsed_seconds > 0:
                    dx = center[0] - previous_center[0]
                    dy = center[1] - previous_center[1]
                    z = self.estimate_depth(bbox[3], frame_height)
                    previous_y_bottom = (
                        previous_position[4]
                        if len(previous_position) > 4
                        else previous_center[1] + previous_height / 2
                    )
                    previous_z = self.estimate_depth(
                        previous_y_bottom,
                        frame_height,
                    )
                    track["velocity_vector"] = {
                        "vx": round(dx / elapsed_seconds, 2),
                        "vy": round(dy / elapsed_seconds, 2),
                        "vz": round((z - previous_z) / elapsed_seconds, 2),
                    }
                    distance = float(np.hypot(
                        dx,
                        dy,
                    ))
                    track["speed_body_lengths_per_second"] = (
                        distance / average_height / elapsed_seconds
                    )
            track["positions"].append(
                (
                    center[0],
                    center[1],
                    max(bbox[3] - bbox[1], 1.0),
                    self.frame_count,
                    bbox[3],
                )
            )

        expired_ids = [
            track_id
            for track_id, track in self.live_tracks.items()
            if (
                self.frame_count - track["last_seen_frame"]
                > config.HISTORY_WINDOW_FRAMES
            )
        ]
        for track_id in expired_ids:
            del self.live_tracks[track_id]

        live_objects = []
        for track_id, track in self.live_tracks.items():
            if track["last_seen_frame"] != self.frame_count:
                continue
            positions = list(track["positions"])
            recent_positions = positions[-5:]
            start, end = recent_positions[0], recent_positions[-1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            speed = track["speed_body_lengths_per_second"]
            previous_locomotion = track["locomotion"]
            if speed < config.IDLE_SPEED_BODY_LENGTHS_PER_SECOND:
                locomotion = "idle"
                direction = "none"
                status = "idle"
            elif speed >= config.RUNNING_SPEED_BODY_LENGTHS_PER_SECOND:
                locomotion = "running"
                direction = (
                    "east" if dx > 0 else "west"
                ) if abs(dx) > abs(dy) else (
                    "south" if dy > 0 else "north"
                )
                status = "moving"
            elif len(positions) > 1:
                locomotion = "walking"
                direction = (
                    "east" if dx > 0 else "west"
                ) if abs(dx) > abs(dy) else (
                    "south" if dy > 0 else "north"
                )
                status = "moving"
            else:
                locomotion = "observed"
                direction = "unknown"
                status = "observed"

            if (
                track["class"] == "person"
                and previous_locomotion == "walking"
                and locomotion == "running"
                and speed >= (
                    track["previous_speed_body_lengths_per_second"]
                    * config.SUDDEN_SPEEDUP_RATIO
                )
            ):
                event = {
                    "type": "sudden_speed_increase",
                    "frame": self.frame_count,
                    "object_id": track_id,
                    "from": "walking",
                    "to": "running",
                    "previous_speed_body_lengths_per_second": round(
                        track["previous_speed_body_lengths_per_second"], 2
                    ),
                    "current_speed_body_lengths_per_second": round(speed, 2),
                    "direction": direction,
                    "evidence": (
                        "Person speed crossed the configured running threshold "
                        "after walking and increased by the configured ratio."
                    ),
                }
                self.recent_behavior_events.append(event)
                print(
                    f"[BEHAVIOR CHANGE] {track_id} changed from walking "
                    f"to running ({event['previous_speed_body_lengths_per_second']:.2f} "
                    f"to {event['current_speed_body_lengths_per_second']:.2f} "
                    "body lengths/second)."
                )
            track["previous_speed_body_lengths_per_second"] = speed
            track["locomotion"] = locomotion

            elapsed_seconds = (
                self.frame_count - track["first_seen_frame"]
            ) / self.fps
            center_x, center_y = positions[-1][:2]
            depth = self.estimate_depth(track["bbox"][3], frame_height)
            normalized_x = round(center_x / frame_width, 4)
            normalized_y = round(center_y / frame_height, 4)
            occupied_zones = [
                zone["name"]
                for zone in self.restricted_zones
                if zone["bbox"][0] <= normalized_x <= zone["bbox"][2]
                and zone["bbox"][1] <= normalized_y <= zone["bbox"][3]
            ]
            live_objects.append({
                "id": track_id,
                "class": track["class"],
                "confidence": round(track["confidence"], 3),
                "position": {
                    "x": round(center_x, 2),
                    "y": round(center_y, 2),
                    "z": depth,
                    "nx": normalized_x,
                    "ny": normalized_y,
                },
                "velocity": track["velocity_vector"].copy(),
                "body_height_px": round(track["bbox"][3] - track["bbox"][1], 2),
                "status": status,
                "locomotion": locomotion,
                "speed_body_lengths_per_second": round(speed, 2),
                "direction": direction,
                "first_seen_frame": track["first_seen_frame"],
                "first_seen_seconds_ago": round(elapsed_seconds, 2),
                "restricted_zones": occupied_zones,
            })
        return live_objects

    def run_reasoning_pipeline(self):
        """Pass changed coordinate data through context, reasoning, and alerts."""
        self.last_reasoning_frame = self.frame_count
        try:
            context = self.context_layer.build_context(
                self.get_json_output(),
                self.current_live_objects,
                list(self.frame_history),
            )
            context["historical_context"]["behavior_events"] = list(
                self.recent_behavior_events
            )
            context["historical_context"]["rule_alerts"] = [
                alert
                for history_frame in self.frame_history
                for alert in history_frame.get("rule_alerts", [])
            ]
            result = self.reasoning_layer.analyze(context)
        except (URLError, TimeoutError, KeyError, ValueError) as error:
            print(f"[LLM ERROR] Could not complete scene reasoning: {error}")
            return

        self.output_layer.emit(
            result,
            frame=self.latest_frame,
            coordinates=self.get_alert_coordinates(result["alerts"]),
            frame_number=self.frame_count,
            camera_id=self.camera_id,
        )

    def clear_cache(self):
        """Forget confirmed and in-progress objects."""
        self.static_objects.clear()
        self.candidate_tracker.clear()
        self.cache_version += 1
        print("[INFO] Static object cache cleared.")
        if config.LLM_ENABLED:
            self.run_reasoning_pipeline()

    def run(self):
        """Main processing loop"""

        if not self.display:
            raise RuntimeError(
                "Headless pipelines are driven by CameraManager, not run()."
            )

        print("\n[CONTROLS]")
        print("  Q = Quit")
        print("  S = Print static data (Text for LLM)")
        print("  J = Print JSON output")
        print("  C = Clear cache\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[INFO] Camera/video stream ended or frame read failed.")
                    break

                self.frame_count += 1
                motion_mask = self.process_frame(frame)

                display_frame = frame.copy()
                if config.SHOW_COORDS:
                    display_frame = self.draw_visualization(display_frame)

                cv2.imshow("Static Environment", display_frame)
                if config.SHOW_MOTION_MASK:
                    cv2.imshow("Motion Mask", motion_mask)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("s"), ord("S")):
                    print(self.get_llm_text_output())
                elif key in (ord("j"), ord("J")):
                    print(self.get_json_output())
                elif key in (ord("c"), ord("C")):
                    self.clear_cache()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """Start the configured static environment system."""
    system = StaticEnvironmentSystem()
    system.run()


if __name__ == "__main__":
    main()