from src.audio.classifier import AudioClassifier

def test_model():

    file_num = input ('''Enter audio file: 
           1) Explosion
           2) Footsteps
           3) Hello
           4) Laser pistol
           5) Rifle
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
    
    classifier = AudioClassifier()
    
    if audio_path: 
        classifier.classify_file(audio_path)
    else: 
        print ("Error with audio file")


if __name__ == "__main__":
    test_model()