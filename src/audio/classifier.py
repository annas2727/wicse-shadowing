from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import librosa

class AudioClassifier:
    def __init__(self):
        print ("Loading model...")
        self.extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        print ("Model loaded.")

    def classify_file(self, audio_path):
        print ("Loading: " + audio_path)

        audio_input, sample_rate = librosa.load(audio_path, sr=16000)

        inputs = self.extractor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        print ("Classifying...")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            top_prediction = torch.argmax(predictions, dim=-1)
            confidence = torch.max(predictions).item()
            
            #get label name
            predicted_label = self.model.config.id2label[top_prediction.item()]

            print(f"✅ Model prediction: Class {predicted_label} with {confidence:.3f} confidence")
            
            top_3 = torch.topk(predictions, 3)
            print("Top 3 predictions:")
            for i, (conf, class_id) in enumerate(zip(top_3.values[0], top_3.indices[0])):
                label = self.model.config.id2label[class_id.item()]
                print(f"  {i+1}. {label}: {conf:.3f}")

            return predicted_label, confidence

