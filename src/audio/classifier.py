from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import librosa
import numpy as np
import os

# -------------------------
# Audio Classifier
# -------------------------
class AudioClassifier:
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", sampling_rate=16000):
        print("Loading model...")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.sampling_rate = sampling_rate
        print("Model loaded.")

    def classify_file(self, audio_path):
        audio_input, _ = librosa.load(audio_path, sr=self.sampling_rate)
        predicted_label, confidence, top3 = self.classify_chunk(audio_input)
        print(f"\n✅ Whole file prediction: {predicted_label} ({confidence:.3f} confidence)")
        print("Top 3 predictions:")
        for i, (lbl, conf) in enumerate(top3, start=1):
            print(f"  {i}. {lbl}: {conf:.3f}")
        return predicted_label, confidence, top3

    def process_long_audio(self, audio_path, segment_duration=2.0, confidence_threshold=0.3):
        audio_input, sr = librosa.load(audio_path, sr=self.sampling_rate)
        chunk_samples = int(segment_duration * sr)
        segments = []

        for i in range(0, len(audio_input), chunk_samples):
            chunk = audio_input[i:i + chunk_samples]
            if len(chunk) < chunk_samples // 2:
                continue
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            timestamp = i / sr
            label, confidence, top3 = self.classify_chunk(chunk)
            if confidence >= confidence_threshold:
                minutes, seconds = divmod(int(timestamp), 60)
                print(f"\nDetected {label} at {minutes:02}:{seconds:02} ({confidence:.3f} confidence)")
                print("Top 3 predictions:")
                for rank, (lbl, conf) in enumerate(top3, start=1):
                    print(f"  {rank}. {lbl}: {conf:.3f}")
                segments.append((timestamp, label, confidence, top3))
        print(f"\nFound {len(segments)} confident audio events!")
        return segments

    def classify_chunk(self, audio_chunk):
        inputs = self.extractor(audio_chunk, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_prediction = torch.argmax(predictions, dim=-1)
            confidence = torch.max(predictions).item()
            predicted_label = self.model.config.id2label[top_prediction.item()]
            top_3 = torch.topk(predictions, 3)
            top3_labels = [(self.model.config.id2label[class_id.item()], conf.item())
                           for conf, class_id in zip(top_3.values[0], top_3.indices[0])]
            return predicted_label, confidence, top3_labels

# -------------------------
# Local “dataset” (file paths + labels)
# -------------------------
dataset = [
    {"path": r"C:/Users/annas/OneDrive/Documents/wicseSP/src/tests/rifle-gun.mp3", "label": "gunfire"},
    {"path": r"C:/Users/annas/OneDrive/Documents/wicseSP/src/tests/game-explosion.mp3", "label": "explosion"},
    {"path": r"C:/Users/annas/OneDrive/Documents/wicseSP/src/tests/apex-long.mp3", "label": "mixed-long-audio"}
]

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    classifier = AudioClassifier()

    print("\nSelect an audio file to classify:")
    for i, entry in enumerate(dataset, start=1):
        print(f"{i}. {os.path.basename(entry['path'])} (expected: {entry['label']})")

    choice = input("Enter the number of the file: ").strip()
    try:
        index = int(choice) - 1
        if 0 <= index < len(dataset):
            selected_file = dataset[index]["path"]
            print(f"\nYou selected: {os.path.basename(selected_file)}")

            # Classify whole file
            classifier.classify_file(selected_file)

            # Process long audio in chunks
            classifier.process_long_audio(selected_file, segment_duration=2.0, confidence_threshold=0.3)
        else:
            print("Invalid selection.")
    except ValueError:
        print("Please enter a valid number.")
