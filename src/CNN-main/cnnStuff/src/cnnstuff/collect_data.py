import librosa
import soundfile as sf
import os
import json
import argparse
import torch
import platform
import subprocess
from .audio_model import AudioCNN
from .predict import predict

def play_audio(file_path):
    """Plays the audio file using a system-specific command."""
    system = platform.system()
    if not os.path.exists(file_path):
        print(f"Warning: Audio file not found at {file_path}")
        return
    print("Playing audio...")
    try:
        if system == "Darwin":
            subprocess.run(["afplay", file_path], check=True, capture_output=True)
        elif system == "Linux":
            subprocess.run(["aplay", file_path], check=True, capture_output=True)
        else:
            print(f"Warning: Unsupported OS '{system}'. Please play the file manually.")
    except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
        print(f"Warning: Could not play audio ({e}). Please play the file manually.")

def get_user_correction(label_map):
    """Gets corrected labels from the user."""
    while True:
        prompt = f"Enter correct label(s) ({', '.join(label_map.keys())}, s=skip): "
        input_str = input(prompt).strip().lower()
        if input_str == 's':
            return None
        try:
            selected_keys = [key.strip() for key in input_str.split(',')]
            if all(key in label_map for key in selected_keys):
                return [label_map[key] for key in selected_keys]
            else:
                print(f"❌ Invalid input.")
        except:
            print(f"❌ Invalid input format.")

def model_assisted_labeling(chunk_files, model, all_labels, threshold):
    """Interactive labeling for new chunks, assisted by the model."""
    # --- Path Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    manual_labels_path = os.path.join(project_root, 'data', 'manual_labels.json')
    audio_dir = os.path.join(project_root, 'data', 'audio_chunks')
    
    label_map = {str(i+1): name for i, name in enumerate(all_labels)}

    try:
        with open(manual_labels_path, 'r') as f:
            existing_labels = json.load(f)
    except FileNotFoundError:
        existing_labels = {}

    unlabeled_chunks = [chunk for chunk in chunk_files if chunk not in existing_labels]
    if not unlabeled_chunks:
        print("All generated audio chunks are already labeled. Nothing to do.")
        return

    print(f"--- Model-Assisted Labeling ---")
    print(f"Found {len(unlabeled_chunks)} new audio chunks to label.")
    
    new_labels = {}
    for i, chunk_file in enumerate(unlabeled_chunks):
        audio_file_path = os.path.join(audio_dir, chunk_file)
        
        # Get both the final prediction and the detailed confidence scores
        predicted_labels, confidence = predict(model, audio_file_path, all_labels, threshold)

        print(f"\n({i+1}/{len(unlabeled_chunks)}) Labeling: {chunk_file}")
        play_audio(audio_file_path)
        
        # --- Show Detailed Model Confidence ---
        print("\n--- Model Confidence ---")
        for label, score in sorted(confidence.items(), key=lambda item: item[1], reverse=True):
            is_predicted = "(Predicted)" if label in predicted_labels else ""
            print(f"- {label:<15}: {score:.4f} {is_predicted}")
        print("------------------------")

        print(f"\nFinal Prediction: {', '.join(predicted_labels) if predicted_labels else 'None'}")

        while True:
            feedback = input("Accept prediction? (y/n/r=replay/q=quit): ").strip().lower()
            if feedback == 'r':
                play_audio(audio_file_path)
                continue
            elif feedback in ['y', 'n', 'q']:
                break
            else:
                print("Invalid input.")

        if feedback == 'q':
            print("Quitting and saving progress...")
            break
        elif feedback == 'y':
            new_labels[chunk_file] = predicted_labels
            print(f"✅ Accepted: {', '.join(predicted_labels)}")
        elif feedback == 'n':
            print("\nPlease provide the correct labels:")
            for key, name in label_map.items():
                print(f"{key} = {name}")
            
            correction = get_user_correction(label_map)
            if correction is not None:
                new_labels[chunk_file] = correction
                print(f"✅ Corrected to: {', '.join(correction)}")
            else:
                print("⏭️  Skipped.")

    existing_labels.update(new_labels)
    with open(manual_labels_path, "w") as f:
        json.dump(existing_labels, f, indent=2)
    
    print(f"\n📊 Results: Added {len(new_labels)} new labels. Saved to {manual_labels_path}")

def split_audio_to_chunks(audio_file, chunk_length=3):
    """Split audio file into 3-second chunks"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    output_dir = os.path.join(project_root, 'data', 'audio_chunks')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading audio file: {audio_file}...")
    audio, sr = librosa.load(audio_file, sr=22050)
    samples_per_chunk = int(chunk_length * sr)
    
    chunks = []
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    for i in range(0, len(audio), samples_per_chunk):
        chunk = audio[i:i + samples_per_chunk]
        if len(chunk) == samples_per_chunk:
            chunk_filename = f"{base_name}_chunk_{i//samples_per_chunk:03d}.wav"
            chunk_path = os.path.join(output_dir, chunk_filename)
            sf.write(chunk_path, chunk, sr)
            chunks.append(chunk_filename)
    
    print(f"Created {len(chunks)} chunks in {output_dir}/")
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split an audio file and label new chunks with model assistance.")
    parser.add_argument("audio_file", type=str, help="Path to the new audio or video file to process.")
    parser.add_argument("--model_path", type=str, default="audio_model.pth", help="Path to the trained model file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold for the model.")
    args = parser.parse_args()

    # --- Path Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    model_path_abs = os.path.join(project_root, args.model_path)
    labels_json_path = os.path.join(project_root, 'data', 'labels.json')

    # --- Load Labels and Model ---
    try:
        with open(labels_json_path, "r") as f:
            all_labels = json.load(f)
    except FileNotFoundError:
        print(f"Error: `{labels_json_path}` not found.")
        exit()

    try:
        model = AudioCNN(num_classes=len(all_labels))
        model.load_state_dict(torch.load(model_path_abs))
    except FileNotFoundError:
        print(f"Error: Model file not found at `{model_path_abs}`. Have you trained a model yet?")
        exit()

    # --- Run Process ---
    if os.path.exists(args.audio_file):
        chunk_filenames = split_audio_to_chunks(args.audio_file)
        model_assisted_labeling(chunk_filenames, model, all_labels, args.threshold)
        print("\nLabeling complete! You can now retrain your model with the new data.")
    else:
        print(f"Error: Audio file not found at '{args.audio_file}'")