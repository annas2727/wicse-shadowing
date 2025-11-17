import torch
import json
import os
import argparse
import subprocess
import platform
from .audio_model import AudioCNN
from .predict import classify 


def play_audio(file_path):
    """Plays the audio file using a system-specific command."""
    system = platform.system()
    
    if not os.path.exists(file_path):
        print(f"Warning: Audio file not found at {file_path}")
        return

    print("Playing audio...")
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", file_path], check=True, capture_output=True)
        elif system == "Linux":
            # Using 'aplay' for WAV files, which is common on Linux
            subprocess.run(["aplay", file_path], check=True, capture_output=True)
        elif system == "Windows":
            # This will open the default media player. It's non-blocking.
            os.startfile(file_path)
        else:
            print(f"Warning: Unsupported OS '{system}'. Please play the file manually.")
    except FileNotFoundError:
        print(f"Warning: Could not find a command-line audio player. Please play the file manually.")
    except (subprocess.CalledProcessError, Exception) as e:
        print(f"Error playing audio: {e}. Please play the file manually.")


def get_user_correction(label_map):
    """Gets corrected labels from the user."""
    while True:
        prompt = f"Enter correct label(s) ({', '.join(label_map.keys())}, s=skip): "
        input_str = input(prompt).strip().lower()

        if input_str == 's':
            return None # Indicates a skip

        try:
            selected_keys = [key.strip() for key in input_str.split(',')]
            if all(key in label_map for key in selected_keys):
                return [label_map[key] for key in selected_keys]
            else:
                print(f"Invalid input. Please use comma-separated numbers from the list.")
        except:
            print(f"Invalid input format. Please try again.")

def interactive_evaluate(model, all_labels, manual_labels_path, audio_dir, threshold):
    """
    Iterate through audio files, show predictions, and ask for user feedback.
    """
    # Load existing manual labels
    try:
        with open(manual_labels_path, 'r') as f:
            manual_labels = json.load(f)
    except FileNotFoundError:
        print(f"Error: Manual labels file not found at {manual_labels_path}")
        return

    # Create the reverse mapping for getting user input
    label_map = {str(i+1): name for i, name in enumerate(all_labels)}

    # Get a list of files to evaluate
    files_to_check = sorted(manual_labels.keys())
    
    print("--- Interactive Evaluation ---")
    print(f"Found {len(files_to_check)} labeled files to evaluate.")
    print("For each file, the audio will play automatically.")
    print("You can then mark the prediction as correct (y) or incorrect (n).")
    print("-" * 40)

    for i, filename in enumerate(files_to_check):
        audio_file_path = os.path.join(audio_dir, filename)
        
        if not os.path.exists(audio_file_path):
            continue

        # Get model's prediction

        predicted_labels, _ = classify(model, audio_file_path, all_labels, threshold)        
        correct_labels = manual_labels.get(filename, [])

        print(f"\n({i+1}/{len(files_to_check)}) Evaluating: {filename}")
        
        # Play the audio
        play_audio(audio_file_path)

        print("-" * 20)
        print(f"Ground Truth: {', '.join(correct_labels)}")
        print(f"Prediction:   {', '.join(predicted_labels) if predicted_labels else 'None'}")
        
        # Ask for user feedback
        while True:
            feedback = input("Is the prediction correct? (y/n/r=replay/q=quit): ").strip().lower()
            if feedback == 'r':
                play_audio(audio_file_path)
            elif feedback in ['y', 'n', 'q']:
                break
            else:
                print("Invalid input. Please enter 'y', 'n', 'r', or 'q'.")

        if feedback == 'q':
            break
        elif feedback == 'n':
            print("\nPlease provide the correct labels:")
            for key, name in label_map.items():
                print(f"{key} = {name}")
            
            new_labels = get_user_correction(label_map)
            if new_labels is not None:
                manual_labels[filename] = new_labels
                print(f"✅ Updated labels for {filename} to: {', '.join(new_labels)}")
            else:
                # If user skips, we remove the label
                del manual_labels[filename]
                print(f"⏭️  Skipped (removed) label for {filename}")

    # Save the updated labels back to the file
    with open(manual_labels_path, 'w') as f:
        json.dump(manual_labels, f, indent=2)
    
    print("\n--- Evaluation Complete ---")
    print(f"💾 Saved updated labels to {manual_labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively evaluate and correct model predictions.")
    parser.add_argument("--model_path", type=str, default="audio_model.pth", help="Path to the trained model file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold (0.0 to 1.0).")
    args = parser.parse_args()

    # --- Path Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    
    model_path_abs = os.path.join(project_root, args.model_path)
    labels_json_path = os.path.join(project_root, 'data', 'labels.json')
    manual_labels_path = os.path.join(project_root, 'data', 'manual_labels.json')
    audio_dir = os.path.join(project_root, 'data', 'audio_chunks')

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

    # --- Run Evaluation ---
    interactive_evaluate(model, all_labels, manual_labels_path, audio_dir, args.threshold)
