import torch
import json
import argparse
import os
from .audio_model import AudioCNN, audio_to_spectrogram

def predict(model, audio_file, all_labels, threshold=0.5):
    """
    Predict the labels for a single audio file and show the model's confidence.
    """
    # Set model to evaluation mode
    model.eval()

    # Process the audio file
    try:
        spectrogram = audio_to_spectrogram(audio_file)
    except Exception as e:
        print(f"Error processing audio file {audio_file}: {e}")
        return [], {}

    # Add a batch dimension (B, C, H, W) before passing to the model
    spectrogram = spectrogram.unsqueeze(0)

    # Get model output (logits)
    with torch.no_grad():
        output = model(spectrogram)
        # Apply sigmoid to convert logits to probabilities (0.0 to 1.0)
        probabilities = torch.sigmoid(output).squeeze(0)

    # --- Decision Making ---
    # The model gives a probability for each sound. We use a threshold to decide
    # whether to include a label in the final prediction.
    predicted_labels = []
    confidence_scores = {}
    for i, prob in enumerate(probabilities):
        label = all_labels[i]
        confidence_scores[label] = prob.item()
        if prob > threshold:
            predicted_labels.append(label)

    return predicted_labels, confidence_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict audio labels from a file.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file to classify.")
    parser.add_argument("--model_path", type=str, default="audio_model.pth", help="Path to the trained model file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold (0.0 to 1.0).")
    args = parser.parse_args()

    # --- Path Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    model_path_abs = os.path.join(project_root, args.model_path)
    labels_json_path = os.path.join(project_root, 'data', 'labels.json')
    audio_file_path_abs = os.path.abspath(args.audio_file)

    # --- Load Labels ---
    try:
        with open(labels_json_path, "r") as f:
            all_labels = json.load(f)
    except FileNotFoundError:
        print(f"Error: `{labels_json_path}` not found.")
        exit()

    # --- Load Model ---
    try:
        model = AudioCNN(num_classes=len(all_labels))
        model.load_state_dict(torch.load(model_path_abs))
    except FileNotFoundError:
        print(f"Error: Model file not found at `{model_path_abs}`.")
        exit()

    # --- Predict ---
    if not os.path.exists(audio_file_path_abs):
        print(f"Error: Audio file not found at `{audio_file_path_abs}`")
        exit()

    predicted, confidence = predict(model, audio_file_path_abs, all_labels, args.threshold)

    print(f"\n--- Prediction for '{os.path.basename(args.audio_file)}' ---")
    print(f"Threshold set to: {args.threshold}\n")

    print("Model Confidence:")
    for label, score in sorted(confidence.items(), key=lambda item: item[1], reverse=True):
        print(f"- {label:<15}: {score:.4f} {'(Predicted)' if label in predicted else ''}")

    print("\nFinal Prediction:")
    if predicted:
        for label in predicted:
            print(f"- {label}")
    else:
        print("No labels predicted with the current threshold.")