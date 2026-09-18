import os
import cv2
import numpy as np
import time
import threading
from collections import deque
from keras.models import model_from_json


emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# --- Absolute paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascades',
                            'haarcascade_frontalface_default.xml')
MODEL_JSON = os.path.join(BASE_DIR, 'model', 'emotion_model.json')
MODEL_H5 = os.path.join(BASE_DIR, 'model', 'emotion_model.h5')

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25
# Smoothing window
SMOOTH_WINDOW = 3
# Face crop trim
FACE_CROP_TRIM = 0.05

# ✅ DEBUG: Set True to see face detection + probability output in the console
DEBUG_PROBS = True


def _empty_result():
    return {
        "emotion_counts": {v: 0 for v in emotion_dict.values()},
        "total_faces": 0,
        "timeline": [],
        "transitions": 0,
        "avg_confidence": 0.0,
        "dominant_emotion": "Neutral",
    }


def _load_model():
    print(f"[INFO] Loading model JSON: {MODEL_JSON}")
    print(f"[INFO] Loading model H5  : {MODEL_H5}")
    json_file = open(MODEL_JSON, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(MODEL_H5)
    return model


def _open_camera(index=0):
    """Try multiple backends to open the webcam."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, test = cap.read()
                if ret and test is not None:
                    print(f"[INFO] Camera opened — index={index}, backend={backend}")
                    return cap
                cap.release()
        except Exception as e:
            print(f"[WARN] Backend {backend} failed: {e}")
    print(f"[ERROR] Could not open camera at index {index} on any backend.")
    return None


def _detect_faces(face_detector, gray):
    """Looser cascade parameters + minSize."""
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50)
    )
    return faces


def _crop_face_tight(gray, x, y, w, h):
    """Safer crop with bounds check."""
    pad_x = int(w * FACE_CROP_TRIM)
    pad_y = int(h * FACE_CROP_TRIM)

    x1 = x + pad_x
    y1 = y + pad_y
    x2 = x + w - pad_x
    y2 = y + h - pad_y

    if x2 - x1 < 40 or y2 - y1 < 40:
        return gray[y:y + h, x:x + w]

    H, W = gray.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    if x2 <= x1 or y2 <= y1:
        return gray[y:y + h, x:x + w]

    return gray[y1:y2, x1:x2]


def _predict_emotion(emotion_model, prob_history, roi_gray):
    """Returns (label, confidence, all_probabilities)."""
    cropped = np.expand_dims(
        np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
    ).astype('float32') / 255.0

    preds = emotion_model.predict(cropped, verbose=0)[0]
    prob_history.append(preds)
    smoothed = np.mean(prob_history, axis=0)

    idx = int(np.argmax(smoothed))
    label = emotion_dict[idx]
    conf = float(smoothed[idx])
    return label, conf, preds


# ============================================================== #
#              BLOCKING MODE (OpenCV native window)              #
# ============================================================== #

def run_emotion_detection(duration=30,
                          smooth_window=SMOOTH_WINDOW,
                          confidence_threshold=CONFIDENCE_THRESHOLD):
    """Blocking emotion detection — opens a native cv2 window."""
    print(f"[INFO] Blocking detection start (duration={duration}s)")

    try:
        emotion_model = _load_model()
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return _empty_result()

    if not os.path.exists(CASCADE_PATH):
        print(f"[ERROR] Haar cascade file NOT FOUND at: {CASCADE_PATH}")
        return _empty_result()

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    if face_detector.empty():
        print(f"[ERROR] Haar cascade could not be loaded.")
        return _empty_result()

    cap = _open_camera(0)
    if cap is None:
        return _empty_result()

    emotion_counts = {v: 0 for v in emotion_dict.values()}
    total_faces = 0
    confidence_sum = 0.0
    confidence_hits = 0
    prob_history = deque(maxlen=smooth_window)
    timeline = []
    last_second_logged = -1
    last_committed_emotion = None
    transitions = 0

    start_time = time.time()
    last_face_count_log = 0.0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Frame grab failed — stopping early.")
            break

        frame = cv2.resize(frame, (850, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = _detect_faces(face_detector, gray)

        # ✅ DEBUG: log face count + brightness once per second
        if DEBUG_PROBS and (time.time() - last_face_count_log) >= 1.0:
            print(f"[DEBUG] Faces: {len(faces)} | "
                  f"Brightness: {gray.mean():.1f}")
            last_face_count_log = time.time()

        dominant_this_frame = None
        conf_this_frame = 0.0

        for (x, y, w, h) in faces:
            roi = _crop_face_tight(gray, x, y, w, h)

            label, conf, preds = _predict_emotion(
                emotion_model, prob_history, roi
            )

            # ✅ DEBUG: raw probabilities
            if DEBUG_PROBS:
                print("  " + " | ".join(
                    f"{emotion_dict[i]}={preds[i]:.2f}" for i in range(7)
                ))

            emotion_counts[label] += 1
            total_faces += 1
            confidence_sum += conf
            confidence_hits += 1

            dominant_this_frame = label
            conf_this_frame = conf

            text_color = (255, 0, 0) if conf >= confidence_threshold \
                else (128, 128, 128)
            cv2.rectangle(frame, (x, y - 50),
                          (x + w, y + h + 10), (0, 255, 0), 4)
            cv2.putText(frame, f"{label} {conf * 100:.1f}%",
                        (x + 5, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        text_color, 2, cv2.LINE_AA)

        cur_sec = int(elapsed)
        if dominant_this_frame and cur_sec != last_second_logged:
            timeline.append({
                "second": cur_sec,
                "emotion": dominant_this_frame,
                "confidence": round(conf_this_frame, 3)
            })
            if (last_committed_emotion is not None and
                    dominant_this_frame != last_committed_emotion):
                transitions += 1
            last_committed_emotion = dominant_this_frame
            last_second_logged = cur_sec

        remaining = max(0, int(duration - elapsed))
        cv2.putText(frame, f"Time left: {remaining}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] User pressed 'q' — stopping early.")
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_conf = (confidence_sum / confidence_hits) if confidence_hits else 0.0
    dominant = (max(emotion_counts, key=emotion_counts.get)
                if total_faces else "Neutral")

    result = {
        "emotion_counts": emotion_counts,
        "total_faces": total_faces,
        "timeline": timeline,
        "transitions": transitions,
        "avg_confidence": round(avg_conf, 3),
        "dominant_emotion": dominant,
    }
    print(f"[INFO] Detection done. Frames={total_faces}, "
          f"dominant={dominant}, transitions={transitions}")
    return result


# ============================================================== #
#                LIVE BROWSER STREAMING SUPPORT                  #
# ============================================================== #

_LIVE_LOCK = threading.Lock()
_LIVE_STATE = {
    'running': False,
    'thread': None,
    'frame': None,
    'emotion': None,
    'confidence': 0.0,
    'elapsed': 0,
    'done': False,
    'final_result': None,
    'duration': 0,
    'stop_flag': False,
    'last_error': None,
}


def _live_worker(duration, smooth_window, confidence_threshold):
    """Background thread for live browser streaming."""
    global _LIVE_STATE
    cap = None

    def _fail(msg):
        print(f"[LIVE-ERROR] {msg}")
        with _LIVE_LOCK:
            _LIVE_STATE['last_error'] = msg

    try:
        try:
            emotion_model = _load_model()
        except Exception as e:
            _fail(f"Model load failed: {e}")
            return

        if not os.path.exists(CASCADE_PATH):
            _fail(f"Cascade file missing at {CASCADE_PATH}")
            return

        face_detector = cv2.CascadeClassifier(CASCADE_PATH)
        if face_detector.empty():
            _fail("Cascade file exists but is corrupted.")
            return

        cap = _open_camera(0)
        if cap is None:
            _fail("Camera could not be opened. Close Zoom/Teams/Camera "
                  "and retry.")
            return

        emotion_counts = {v: 0 for v in emotion_dict.values()}
        total_faces = 0
        confidence_sum = 0.0
        confidence_hits = 0
        prob_history = deque(maxlen=smooth_window)
        timeline = []
        last_second_logged = -1
        last_committed_emotion = None
        transitions = 0

        start_time = time.time()
        last_face_count_log = 0.0

        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration or _LIVE_STATE['stop_flag']:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame = cv2.resize(frame, (850, 720))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = _detect_faces(face_detector, gray)

            # ✅ DEBUG: face count + brightness every second
            if DEBUG_PROBS and (time.time() - last_face_count_log) >= 1.0:
                print(f"[DEBUG] Faces: {len(faces)} | "
                      f"Brightness: {gray.mean():.1f}")
                last_face_count_log = time.time()

            dominant_this_frame = None
            conf_this_frame = 0.0

            for (x, y, w, h) in faces:
                roi = _crop_face_tight(gray, x, y, w, h)

                label, conf, preds = _predict_emotion(
                    emotion_model, prob_history, roi
                )

                # ✅ DEBUG: raw probabilities
                if DEBUG_PROBS:
                    print("  " + " | ".join(
                        f"{emotion_dict[i]}={preds[i]:.2f}" for i in range(7)
                    ))

                emotion_counts[label] += 1
                total_faces += 1
                confidence_sum += conf
                confidence_hits += 1

                dominant_this_frame = label
                conf_this_frame = conf

                text_color = (255, 0, 0) if conf >= confidence_threshold \
                    else (128, 128, 128)
                cv2.rectangle(frame, (x, y - 50),
                              (x + w, y + h + 10), (0, 255, 0), 4)
                cv2.putText(frame, f"{label} {conf * 100:.1f}%",
                            (x + 5, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            text_color, 2, cv2.LINE_AA)

            cur_sec = int(elapsed)
            if dominant_this_frame and cur_sec != last_second_logged:
                timeline.append({
                    "second": cur_sec,
                    "emotion": dominant_this_frame,
                    "confidence": round(conf_this_frame, 3)
                })
                if (last_committed_emotion is not None and
                        dominant_this_frame != last_committed_emotion):
                    transitions += 1
                last_committed_emotion = dominant_this_frame
                last_second_logged = cur_sec

            remaining = max(0, int(duration - elapsed))
            cv2.putText(frame, f"Time left: {remaining}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)

            ok, jpg = cv2.imencode('.jpg', frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with _LIVE_LOCK:
                    _LIVE_STATE['frame'] = jpg.tobytes()
                    _LIVE_STATE['emotion'] = dominant_this_frame
                    _LIVE_STATE['confidence'] = conf_this_frame
                    _LIVE_STATE['elapsed'] = round(elapsed, 2)

        avg_conf = (confidence_sum / confidence_hits) if confidence_hits else 0.0
        dominant = (max(emotion_counts, key=emotion_counts.get)
                    if total_faces else "Neutral")

        with _LIVE_LOCK:
            _LIVE_STATE['final_result'] = {
                "emotion_counts": emotion_counts,
                "total_faces": total_faces,
                "timeline": timeline,
                "transitions": transitions,
                "avg_confidence": round(avg_conf, 3),
                "dominant_emotion": dominant,
                "duration": duration,
            }
            _LIVE_STATE['done'] = True

        print(f"[LIVE] Done. Frames={total_faces}, dominant={dominant}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _fail(f"Worker crashed: {e}")

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        with _LIVE_LOCK:
            _LIVE_STATE['running'] = False
            _LIVE_STATE['stop_flag'] = False
        print("[LIVE] Worker exited — state reset.")


# ----------------- Public API used by app.py ----------------- #

def start_session(duration, user_id=None):
    """Start a live detection session in a background thread."""
    global _LIVE_STATE
    with _LIVE_LOCK:
        if _LIVE_STATE['running']:
            thread = _LIVE_STATE.get('thread')
            if thread is None or not thread.is_alive():
                print("[LIVE] Zombie session detected — resetting.")
                _LIVE_STATE['running'] = False
                _LIVE_STATE['stop_flag'] = False
            else:
                return False

        _LIVE_STATE.update({
            'running': True,
            'frame': None,
            'emotion': None,
            'confidence': 0.0,
            'elapsed': 0,
            'done': False,
            'final_result': None,
            'duration': duration,
            'stop_flag': False,
            'last_error': None,
        })
        t = threading.Thread(
            target=_live_worker,
            args=(duration, SMOOTH_WINDOW, CONFIDENCE_THRESHOLD),
            daemon=True
        )
        _LIVE_STATE['thread'] = t
        t.start()
    return True


def get_frame():
    with _LIVE_LOCK:
        return _LIVE_STATE['frame']


def get_status():
    with _LIVE_LOCK:
        return {
            'emotion': _LIVE_STATE['emotion'],
            'confidence': _LIVE_STATE['confidence'],
            'elapsed': _LIVE_STATE['elapsed'],
            'done': _LIVE_STATE['done'],
            'running': _LIVE_STATE['running'],
            'error': _LIVE_STATE.get('last_error'),
        }


def stop_session():
    with _LIVE_LOCK:
        _LIVE_STATE['stop_flag'] = True


def is_session_active():
    with _LIVE_LOCK:
        return _LIVE_STATE['running']


def get_final_result():
    with _LIVE_LOCK:
        result = _LIVE_STATE['final_result']
        _LIVE_STATE['final_result'] = None
        _LIVE_STATE['frame'] = None
        _LIVE_STATE['emotion'] = None
        return result
