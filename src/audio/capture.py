import sys
import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import json
import time

CNN_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "CNNmain", "cnnStuff", "src")
)
print("Adding CNN PATH:", CNN_PATH)
sys.path.insert(0, CNN_PATH)

from cnnstuff.predict import predict
from direction import detect_direction

SAMPLE_RATE = 48000
CHUNK_DURATION = 1 # in seconds
CHANNELS = 8      
DEVICE_INDEX = 1  # set automatically later

CHUNKS_DIR = "data/audio_chunks"

def write_json(json_obj, path="latest_direction.json"):
        tmp = path + ".tmp"
        
        # write temporary
        with open(tmp, "w") as f:
            json.dump(json_obj, f)

        # retry replace 10 times if overlay temporarily locks file
        for _ in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.01)  # wait 10ms

        print("WARNING: Could not replace JSON file due to file lock.")

def find_vbcable():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        name = d["name"].lower()
        if "cable output" in name or "vb-audio" in name:
            return i
    raise RuntimeError("VB-Cable device not found. Is it installed?")

def record_chunk(filename):
    print(f"Recording {filename}...")
    audio = sd.rec(
        int(CHUNK_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS, #8 channels for 7.1, some will be blank if stereo or 5.1
        device=DEVICE_INDEX
    )
    sd.wait()

    # Convert float32 into int16 WAV
    audio_int16 = (audio * 32767).astype(np.int16)
    write(filename, SAMPLE_RATE, audio_int16)

def run_prediction(filepath):

    print(f"Predicting {filepath}...")

    predicted, confidence = predict(filepath, threshold=0.3)
    direction = detect_direction(filepath)

    # display output
    print("\n--- MODEL PREDICTION ---")
    for label, score in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
        print(f"{label:12} {score:.4f}{'  (PRED)' if label in predicted else ''}")

    print("\nFinal Predicted Labels:", predicted)
    print(f"Direction: {direction['angle']:.1f}°")
    print(f"Intensity: {direction['intensity']:.3f}")
    print("-" * 50)

    # WRITE JSON for overlay and use tmp so it never reads a half written file
    write_json({
        "angle": direction["angle"],
        "intensity": direction["intensity"],
        "label": predicted,
        "confidence": confidence
    })

    
if __name__ == "__main__":
    DEVICE_INDEX = find_vbcable()
    print(f"Using VB-Cable device index: {DEVICE_INDEX}")

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    chunk_id = 0
    print("Starting LIVE audio classifier...")

    while True:
        filename = os.path.join(CHUNKS_DIR, f"live_chunk_{chunk_id:04}.wav")
        record_chunk(filename)
        run_prediction(filename)
        chunk_id += 1
        time.sleep(0.2)  # small gap
