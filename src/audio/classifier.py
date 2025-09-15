from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch

class AudioClassifier:
    def __init__(self):
        print ("Loading model...")
        self.extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        print ("Model loaded.")

    def classify_file(self, audio_path):
        print ("Loading: " + audio_path)

        audio_input, sample_rate = librosa.load(audio_path)

        inputs = self.extractor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        print ("Classifying...")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Show top result
            top_prediction = torch.argmax(predictions, dim=-1)
            confidence = torch.max(predictions).item()
            
            print(f"✅ Model prediction: Class {top_prediction.item()} with {confidence:.3f} confidence")
            
            return top_prediction.item(), confidence

