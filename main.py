from src.audio.classifier import AudioClassifier
import librosa
def test_model():

    file_num = input ('''Enter audio file: 
           1) Explosion
           2) Footsteps
           3) Hello
           4) Laser pistol
           5) Rifle
           6) Long audio (Apex Legends)
    ''')

    audio_path = None
    if (file_num == "1"):
        audio_path = "src/tests/game-explosion.mp3"
    elif (file_num == "2"):
        audio_path = "src/tests/footsteps.mp3"     
    elif (file_num == "3"):
        audio_path = "src/tests/hello.mp3"
    elif (file_num == "4"):
        audio_path = "src/tests/laser-pistol.mp3"              
    elif (file_num == "5"):
        audio_path = "src/tests/rifle-gun.mp3"
    elif (file_num == "6"):
        audio_path = "src/tests/apex-long.mp3"
    
    classifier = AudioClassifier()
    
    if audio_path: 
        try: 
            full_audio, sample_rate = librosa.load(audio_path, sr=16000)
            duration = len(full_audio) / sample_rate
            print(f"📊 Audio duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
            if duration > 30: 
                print ("Splitting long audio into chunks...")
                classifier.process_long_audio(audio_path, segment_duration=2.0, confidence_threshold=0.3)
            else: 
                classifier.classify_file(audio_path)
        except Exception as e: 
            print ("Error loading audio file: " + str(e))
    else: 
        print ("Error with audio file")


if __name__ == "__main__":
    test_model()