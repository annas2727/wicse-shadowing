import torch
import json
import os
from .audio_model import AudioCNN, audio_to_spectrogram

# -------------------------------------------------------------
# INTERNAL: classify audio given a loaded model and labels
# -------------------------------------------------------------
def classify(model, audio_file, labels, threshold=0.5):

    model.eval()

    # Convert audio → spectrogram
    spectrogram = audio_to_spectrogram(audio_file)
    spectrogram = spectrogram.unsqueeze(0)

    # Get probabilities
    with torch.no_grad():
        logits = model(spectrogram)
        probs = torch.sigmoid(logits).squeeze(0)

    predicted = []
    confidence = {}

    for i, p in enumerate(probs):
        label = labels[i]
        p_val = float(p.item())
        confidence[label] = p_val
        if p_val > threshold:
            predicted.append(label)

    return predicted, confidence

# PUBLIC API: simple function capture.py can call
def predict(filepath, threshold=0.5):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cnnmain_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    model_path  = os.path.join(cnnmain_root, "audio_model.pth")
    labels_path = os.path.join(cnnmain_root, "data", "labels.json")

    with open(labels_path, "r") as f:
        labels = json.load(f)

    # Load model
    model = AudioCNN(num_classes=len(labels))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Classify
    predicted, confidence = classify(model, filepath, labels, threshold)

    return predicted, confidence


# OLD INTERFACE (OPTIONAL)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict audio labels")
    parser.add_argument("audio_file")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predicted, confidence = predict(args.audio_file, args.threshold)

    print("\n--- Prediction ---\n")
    for label, score in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
        print(f"{label:12} {score:.4f}{'  (Predicted)' if label in predicted else ''}")

    print("\nFinal:", predicted if predicted else "None")
