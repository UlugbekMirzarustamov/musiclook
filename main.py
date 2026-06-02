"""
MusicLook
Look down → music plays. Look up → music pauses.

Webcam → MediaPipe FaceMesh → head-pitch estimate → pygame audio control.
"""

import cv2
import mediapipe as mp
import pygame
import sys
import os
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CAMERA_INDEX     = 0
GAZE_MODE        = "eye"   # "eye" (look down with your eyes) or "head" (whole-head tilt)
SHOW_LANDMARKS   = False

# Head-pitch mode (whole-head tilt)
HEAD_DOWN_THRESH = 0.10

# Eye-gaze mode (iris position between eyelids)
EYE_DOWN_THRESH  = 0.07    # how far below neutral the iris must drop to trigger
EYE_NEUTRAL      = 0.50    # iris vertical ratio when looking straight ahead
EYE_SMOOTHING    = 0.4     # 0..1, higher = snappier but jitterier

# Find the music file next to this script (music.mp3 preferred, music.wav fallback)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_music() -> str:
    for name in ("music.mp3", "music.wav"):
        path = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(path):
            return path
    return os.path.join(SCRIPT_DIR, "music.mp3")


MUSIC_FILE = _find_music()
# ──────────────────────────────────────────────────────────────────────────────

# ─── Colors (BGR for OpenCV) ───────────────────────────────────────────────────
BLACK      = (0,   0,   0)
WHITE      = (255, 255, 255)
RED        = (0,   0,   220)
GREEN      = (0,   200, 60)
ORANGE     = (0,   140, 255)
GRAY_DARK  = (25,  25,  25)
GRAY_MID   = (60,  60,  60)
# ──────────────────────────────────────────────────────────────────────────────


def init_audio(music_path: str) -> bool:
    """Initialize pygame mixer and load music. Returns True on success."""
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    if not os.path.exists(music_path):
        print(f"[WARN] Music file not found: '{music_path}'")
        print("       Drop any .mp3 into this folder and rename it music.mp3")
        return False
    try:
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.set_volume(0.85)
        return True
    except Exception as e:
        print(f"[WARN] Could not load audio: {e}")
        return False


def get_head_pitch(face_landmarks, img_h: int, img_w: int) -> float:
    """
    Whole-head pitch:
      positive  → head tilted DOWN
      negative  → head tilted UP
    Uses the vertical offset between nose tip (#4) and forehead (#10),
    normalised by face height so it's scale-invariant.
    """
    nose_tip = face_landmarks.landmark[4]
    forehead = face_landmarks.landmark[10]
    chin     = face_landmarks.landmark[152]

    nose_y     = nose_tip.y * img_h
    forehead_y = forehead.y * img_h
    chin_y     = chin.y     * img_h

    face_height = chin_y - forehead_y + 1e-6
    # positive pitch = looking DOWN (chin toward chest)
    pitch = (nose_y - forehead_y) / face_height - 0.42  # 0.42 = calibrated neutral
    return pitch


def get_eye_gaze(face_landmarks) -> float:
    """
    Eye-gaze pitch from iris position (needs refine_landmarks=True):
      positive  → eyes looking DOWN
      negative  → eyes looking UP
    For each eye, measure where the iris centre sits vertically between the
    upper and lower eyelid (0 = touching upper lid, 1 = touching lower lid).
    Averaged across both eyes, then offset by the neutral ratio.
    """
    lm = face_landmarks.landmark

    def eye_ratio(iris_center: int, upper_lid: int, lower_lid: int) -> float:
        iris_y  = lm[iris_center].y
        upper_y = lm[upper_lid].y
        lower_y = lm[lower_lid].y
        return (iris_y - upper_y) / ((lower_y - upper_y) + 1e-6)

    # Right eye: iris centre 468, upper lid 159, lower lid 145
    # Left eye:  iris centre 473, upper lid 386, lower lid 374
    right = eye_ratio(468, 159, 145)
    left  = eye_ratio(473, 386, 374)
    ratio = (right + left) / 2.0

    # positive = looking DOWN (iris toward lower lid)
    return ratio - EYE_NEUTRAL


def draw_hud(frame, pitch: float, threshold: float,
             playing: bool, has_audio: bool, no_face_frames: int,
             mode: str = "eye"):
    """Overlay status info on the frame."""
    h, w = frame.shape[:2]

    # ── semi-transparent bottom bar ──────────────────────────────────────────
    bar_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), GRAY_DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Pitch gauge ───────────────────────────────────────────────────────────
    gauge_x, gauge_y = 20, h - bar_h + 15
    gauge_w, gauge_h = 200, 20
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), GRAY_MID, -1)
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), WHITE, 1)

    # Fill bar proportional to pitch (clamp -0.50 … +0.50)
    clamped    = max(-0.50, min(0.50, pitch))
    fill_ratio = (clamped + 0.50) / 1.00
    bar_color  = RED if pitch >= threshold else GREEN
    fill_px    = int(fill_ratio * gauge_w)
    cv2.rectangle(frame, (gauge_x, gauge_y),
                  (gauge_x + fill_px, gauge_y + gauge_h), bar_color, -1)

    # Threshold marker
    thresh_px = gauge_x + int(((threshold + 0.50) / 1.00) * gauge_w)
    cv2.line(frame, (thresh_px, gauge_y - 3), (thresh_px, gauge_y + gauge_h + 3), ORANGE, 2)

    metric = "Gaze" if mode == "eye" else "Pitch"
    cv2.putText(frame, f"{metric}: {pitch:+.3f}  thresh: {threshold:+.3f}  [{mode.upper()}]  (M to switch)",
                (gauge_x, gauge_y + gauge_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    # ── State label ──────────────────────────────────────────────────────────
    down_txt = "EYES DOWN" if mode == "eye" else "HEAD DOWN"
    up_txt   = "EYES ON CAMERA" if mode == "eye" else "HEAD UP"
    if no_face_frames > 15:
        label, color = "NO FACE DETECTED", ORANGE
    elif pitch >= threshold:
        label, color = f"{down_txt}  ▼  MUSIC ON ♫", RED
    else:
        label, color = f"{up_txt}  ▲  PAUSED", GREEN

    cv2.putText(frame, label, (gauge_x, h - bar_h + 75),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    # ── Audio warning ────────────────────────────────────────────────────────
    if not has_audio:
        cv2.putText(frame, "[NO AUDIO - see terminal]",
                    (w - 280, h - bar_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1)

    # ── Corner badge ─────────────────────────────────────────────────────────
    cv2.putText(frame, "MUSICLOOK", (w - 130, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, RED, 2)

    return frame


def main():
    mode      = GAZE_MODE if GAZE_MODE in ("eye", "head") else "eye"
    threshold = EYE_DOWN_THRESH if mode == "eye" else HEAD_DOWN_THRESH

    print("=" * 50)
    print("  MUSICLOOK")
    print("  Look DOWN -> music plays | Look UP -> paused")
    print(f"  Mode: {mode.upper()}   (press M to switch eye/head)")
    print(f"  Threshold: {threshold:+.3f}")
    print("=" * 50)

    # ── Audio setup ───────────────────────────────────────────────────────────
    pygame.init()
    has_audio = init_audio(MUSIC_FILE)

    # ── MediaPipe FaceMesh ────────────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils if SHOW_LANDMARKS else None

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {CAMERA_INDEX}.")
        print("        Try changing CAMERA_INDEX = 1 at the top of main.py")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[INFO] Running - press Q to quit, M to switch eye/head mode.\n")

    playing        = False
    no_face_frames = 0
    pitch          = 0.0
    eye_smoothed   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, retrying...")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)   # mirror for natural feel
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            no_face_frames = 0
            fl = results.multi_face_landmarks[0]

            if SHOW_LANDMARKS and mp_draw:
                mp_draw.draw_landmarks(
                    frame, fl,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_draw.DrawingSpec(color=(0, 180, 0), thickness=1, circle_radius=1),
                    mp_draw.DrawingSpec(color=(0, 100, 0), thickness=1),
                )

            if mode == "eye":
                raw          = get_eye_gaze(fl)
                eye_smoothed = EYE_SMOOTHING * raw + (1 - EYE_SMOOTHING) * eye_smoothed
                pitch        = eye_smoothed
            else:
                pitch = get_head_pitch(fl, h, w)

            if pitch >= threshold:
                # ── LOOK DOWN (chin toward chest) → play ─────────────────
                if not playing and has_audio:
                    if pygame.mixer.music.get_pos() == -1:
                        pygame.mixer.music.play(-1)
                    else:
                        pygame.mixer.music.unpause()
                    playing = True
                    print(f"[>] Playing  (pitch={pitch:+.3f})")
            else:
                # ── LOOKING AT CAMERA → pause ─────────────────────────────
                if playing and has_audio:
                    pygame.mixer.music.pause()
                    playing = False
                    print(f"[#] Paused   (pitch={pitch:+.3f})")
        else:
            no_face_frames += 1
            if no_face_frames > 15 and playing and has_audio:
                pygame.mixer.music.pause()
                playing = False

        frame = draw_hud(frame, pitch, threshold,
                         playing, has_audio, no_face_frames, mode)

        cv2.imshow("MusicLook", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            # toggle eye/head mode; pause and reset so state stays sane
            mode      = "head" if mode == "eye" else "eye"
            threshold = EYE_DOWN_THRESH if mode == "eye" else HEAD_DOWN_THRESH
            eye_smoothed = 0.0
            if playing and has_audio:
                pygame.mixer.music.pause()
                playing = False
            print(f"[~] Switched to {mode.upper()} mode (thresh={threshold:+.3f})")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if has_audio:
        pygame.mixer.music.stop()
    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("\n[INFO] Bye!\n")


if __name__ == "__main__":
    main()
