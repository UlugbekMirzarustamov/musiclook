"""
generate_tone.py — make a royalty-free placeholder riff (music.wav) for testing.
No external deps beyond numpy. Run:  python generate_tone.py
"""

import os
import wave
import numpy as np

SR        = 44100
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT       = os.path.join(SCRIPT_DIR, "music.wav")

# A driving little riff in E (note, beats)
BPM   = 140
BEAT  = 60.0 / BPM
NOTES = [82.41, 82.41, 98.00, 110.00, 82.41, 110.00, 98.00, 82.41]  # E2 G2 A2 ...


def synth_note(freq, dur):
    """Distorted saw -> sounds vaguely like a power chord."""
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    # saw wave + a fifth on top for a chord-y feel
    saw  = 2 * (t * freq - np.floor(0.5 + t * freq))
    saw5 = 2 * (t * freq * 1.5 - np.floor(0.5 + t * freq * 1.5))
    wave_ = 0.6 * saw + 0.4 * saw5
    wave_ = np.tanh(2.2 * wave_)                       # soft clip = distortion
    env   = np.minimum(1.0, np.exp(-3.0 * t) + 0.25)   # pluck envelope
    return wave_ * env


def main():
    riff = np.concatenate([synth_note(f, BEAT) for f in NOTES])
    riff = np.tile(riff, 4)                              # ~13s loop
    riff /= np.max(np.abs(riff))                         # normalise
    audio = (riff * 0.85 * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio]).flatten()  # duplicate to 2 channels

    with wave.open(OUT, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())

    print(f"[OK] Wrote {OUT}  ({len(riff)/SR:.1f}s loop)")


if __name__ == "__main__":
    main()
