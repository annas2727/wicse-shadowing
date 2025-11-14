import sys
import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import subprocess
import time
import json

from CNNmain.cnnStuff.src.cnnstuff.predict import predict
from audio.direction import detect_direction

SAMPLE_RATE = 48000
CHUNK_DURATION = 5 # in seconds
CHANNELS = 6         # stereo (VB Cable)
DEVICE_INDEX = 1  # set automatically later

CHUNKS_DIR = "data/audio_chunks"

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

    # Convert float32 → int16 WAV
    audio_int16 = (audio * 32767).astype(np.int16)
    write(filename, SAMPLE_RATE, audio_int16)

def run_prediction(filepath):
    print(f"Predicting {filepath}...")
    result = subprocess.run(
        ["poetry", "run", "python", "-m", "cnnstuff.predict",
        "--threshold", "0.3",
        filepath],
        capture_output=True,
        text=True
    )
    
    print("MODEL OUTPUT:")
    output = result.stdout.strip()
    if not output:
        print("(no prediction — maybe silence?)")
    else:
        print(output)
    print("-" * 50)

    direction = detect_direction(filepath)
    print(f"Detected direction: {direction.get('angle', 'N/A'):.1f}° with intensity {direction.get('intensity', 0):.3f}")
    print(f"Raw energies: {direction.get('raw_energies', {})}")


def run_prediction(filepath):
    # existing classification
    print(f"Predicting {filepath}...")
    predicted, confidence = predict(filepath)

    direction = detect_direction(filepath)

    # write direction + prediction to shared JSON
    with open("latest_direction.json", "w") as f:
        json.dump({
            "angle": direction["angle"],
            "intensity": direction["intensity"],
            "label": predicted,
            "confidence": confidence
        }, f)

if __name__ == "__main__":
    DEVICE_INDEX = find_vbcable()
    print(f"Using VB-Cable device index: {DEVICE_INDEX}")

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    chunk_id = 0
    print("Starting LIVE audio classifier... press CTRL+C to stop.")

    while True:
        filename = os.path.join(CHUNKS_DIR, f"live_chunk_{chunk_id:04}.wav")
        record_chunk(filename)
        run_prediction(filename)
        chunk_id += 1
        time.sleep(0.2)  # small gap
