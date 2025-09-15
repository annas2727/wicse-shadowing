import numpy as np
import pyaudio 

from src.overlay.display import Overlay

# Load model directly

extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

#instantiate py audio
audio = pyaudio.PyAudio()

#open stream
stream = audio.open(format=audio.get_format_from_width(wf.get))
    
with wave.open(sys.argv[1], 'rb') as wf:
    # Instantiate PyAudio and initialize PortAudio system resources (1)
    audio = pyaudio.PyAudio()

    # Open stream (2)
    stream = audio.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
     # Play samples from the wave file (3)
    while len(data := wf.readframes(CHUNK)):  # Requires Python 3.8+ for :=
        stream.write(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Show top result
    top_prediction = torch.argmax(predictions, dim=-1)
    confidence = torch.max(predictions).item()
    
    print(f"✅ Model prediction: Class {top_prediction.item()} with {confidence:.3f} confidence")
    
    print("\n🎉 SUCCESS! Everything is working!")
    print("\nNext step: We'll add real audio capture")
    

if __name__ == "__main__":
    app = Overlay()
    app.run()