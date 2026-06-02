# MusicLook

Look **down** and music plays. Look back **up** and it pauses.

Your webcam feeds [MediaPipe FaceMesh](https://developers.google.com/mediapipe), which gives facial landmarks every frame. Two ways to trigger:

- **Eye mode** (default) — uses the **iris** landmarks to measure where your eyes point. Glance down with just your eyes and the music plays; your head can stay still.
- **Head mode** — uses whole-head **pitch** from the nose/forehead/chin landmarks. Tilt your head down to play.

Cross the threshold and `pygame` starts the music; look back and it pauses. A live HUD shows the gauge, threshold, current state, and active mode. Press **M** anytime to switch modes.

Inspired by [RockLook](https://github.com/Botirsherov/RockLook).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Drop any `.mp3` into this folder and rename it `music.mp3`.

## Run

```powershell
python main.py
```

- Glance **down** with your eyes → music plays (eye mode)
- Look back at the camera → music pauses
- Press **M** to switch eye ↔ head mode
- Press **Q** to quit

## Tuning

Edit the constants at the top of `main.py`:

| Constant           | Default | Meaning                                                       |
|--------------------|---------|---------------------------------------------------------------|
| `CAMERA_INDEX`     | `0`     | Webcam index — try `1` if the wrong camera opens              |
| `GAZE_MODE`        | `"eye"` | Start in `"eye"` (iris) or `"head"` (whole-head tilt) mode    |
| `EYE_DOWN_THRESH`  | `0.07`  | How far the iris drops below neutral to trigger (eye mode)    |
| `EYE_NEUTRAL`      | `0.50`  | Iris vertical ratio when looking straight ahead              |
| `EYE_SMOOTHING`    | `0.4`   | 0–1; higher = snappier but jitterier iris tracking           |
| `HEAD_DOWN_THRESH` | `0.10`  | Pitch needed to trigger playback (head mode)                 |
| `SHOW_LANDMARKS`   | `False` | Draw the FaceMesh overlay for debugging                      |

Eye tracking is per-person — watch the on-screen **Gaze** value: it should hover near `0.000` looking straight ahead and rise when you glance down. Set `EYE_DOWN_THRESH` a bit below the value you see when looking down comfortably. The head-mode neutral offset (`0.42` in `get_head_pitch`) is calibrated for a roughly head-on camera.

## How it works

1. **Sensor** — webcam frames via OpenCV.
2. **Detector** — MediaPipe FaceMesh → 478 landmarks (incl. iris, via `refine_landmarks=True`).
3. **Comparator** —
   - *eye mode:* `get_eye_gaze()` measures each iris's vertical position between the eyelids, averages both eyes, and compares to `EYE_DOWN_THRESH`.
   - *head mode:* `get_head_pitch()` turns nose/forehead/chin into a scale-invariant pitch vs `HEAD_DOWN_THRESH`.
4. **Actuator** — `pygame.mixer` plays / pauses `music.mp3`.

If no face is seen for ~15 frames, playback pauses automatically.
