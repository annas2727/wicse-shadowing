from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import librosa
import numpy as np

class AudioClassifier:
    def __init__(self, 
                model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                augment_prob=0.8,
                sampling_rate=16000):
        print ("Loading model...")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.sampling_rate = sampling_rate
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

    def process_long_audio(self, audio_path, segment_duration=2.0, confidence_threshold=0.3):
        print("Processing long audio: " + audio_path)
        
        audio_input, sample_rate = librosa.load(audio_path, sr=self.sampling_rate)
        chunk_samples = int(segment_duration * sample_rate)

        segments = []
        for i in range(0, len(audio_input), chunk_samples):
            chunk = audio_input[i:i + chunk_samples]

            if len(chunk) < chunk_samples // 2: 
                continue

            if len(chunk) < chunk_samples:
                padding = np.zeros(chunk_samples - len(chunk))
                chunk = np.concatenate((chunk, padding))

            timestamp = i / sample_rate
            label, confidence, top3_labels = self.classify_chunk(chunk)  # now returns 3 values

            if confidence >= confidence_threshold:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)   
                
                print(f"\nDetected {label} at {minutes:02}:{seconds:02} with {confidence:.3f} confidence")
                print("Top 3 predictions:")
                for rank, (lbl, conf) in enumerate(top3_labels, start=1):
                    print(f"  {rank}. {lbl}: {conf:.3f}")

                # save all info for later use
                segments.append((timestamp, label, confidence, top3_labels))
        
        print(f"\n🎉 Found {len(segments)} confident audio events!")
        return segments

        
    def classify_chunk(self, audio_chunk):
        inputs = self.extractor(audio_chunk, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            top_prediction = torch.argmax(predictions, dim=-1)
            confidence = torch.max(predictions).item()
            predicted_label = self.model.config.id2label[top_prediction.item()]

            # top 3 predictions
            top_3 = torch.topk(predictions, 3)
            top3_labels = [(self.model.config.id2label[class_id.item()], conf.item())
                        for conf, class_id in zip(top_3.values[0], top_3.indices[0])]

            return predicted_label, confidence, top3_labels