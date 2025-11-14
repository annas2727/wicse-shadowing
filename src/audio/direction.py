import numpy as np
import soundfile as sf

#captures directional audio as 7.1, but only uses the first two channels for direction detection

def detect_direction(audio_chunk):

    audio_chunk, samplerate = sf.read(audio_chunk)

    if audio_chunk.ndim == 1:
        # Mono: duplicate to 8 channels
        audio_chunk = np.stack([audio_chunk] * 8, axis=1)

    num_channels = audio_chunk.shape[1]

    #get num of channels 
    if num_channels == 2: 
        FL = audio_chunk[:, 0]
        FR = audio_chunk[:, 1]

        energies = {
            "FL": float(np.sum(FL**2)),
            "FR": float(np.sum(FR**2))
        }

        energy_left = energies["FL"]
        energy_right = energies["FR"]
        energy_front = energy_left + energy_right
        energy_back = 0


    elif num_channels == 6:
        #map channels
        FL = audio_chunk[:, 0] 
        FR = audio_chunk[:, 1]
        C = audio_chunk[:, 2]
        LFE = audio_chunk[:, 3] 
        SL = audio_chunk[:, 4]
        SR = audio_chunk[:, 5]

        #compute energy
        energies = {
            "FL": float(np.sum(FL**2)),
            "FR": float(np.sum(FR**2)),
            "C":  float(np.sum(C**2)),
            "LFE":float(np.sum(LFE**2)),
            "SL": float(np.sum(SL**2)),
            "SR": float(np.sum(SR**2)),
        }
    
        energy_left  = energies["FL"] + energies["SL"]
        energy_right = energies["FR"] + energies["SR"]
        energy_front = energies["FL"] + energies["FR"] + energies["C"]
        energy_back  = energies["SL"] + energies["SR"]

    elif num_channels >= 8:
        #map channels
        FL = audio_chunk[:, 0] #if stereo, only this and FR
        FR = audio_chunk[:, 1]
        C = audio_chunk[:, 2]
        LFE = audio_chunk[:, 3] #bass, non directional 
        SL = audio_chunk[:, 4]
        SR = audio_chunk[:, 5]
        RL = audio_chunk[:, 6]
        RR = audio_chunk[:, 7]

        #compute energy
        energies = {
            "FL": float(np.sum(FL**2)),
            "FR": float(np.sum(FR**2)),
            "C":  float(np.sum(C**2)),
            "LFE":float(np.sum(LFE**2)),
            "SL": float(np.sum(SL**2)),
            "SR": float(np.sum(SR**2)),
            "RL": float(np.sum(RL**2)),
            "RR": float(np.sum(RR**2)),
        }
        
        #compute energy regions
        #add all the left, right, front, back to get total energy region
        energy_left = np.sum(FL**2) + np.sum(SL**2) + np.sum(RL**2)
        energy_right = np.sum(FR**2) + np.sum(SR**2) + np.sum(RR**2)
        energy_front = np.sum(FL**2) + np.sum(FR**2) + np.sum(C**2)
        energy_back = np.sum(SL**2) + np.sum(SR**2)

    else: 
        raise ValueError(f"Unsupported number of channels: {num_channels}")

    
    #calculate angle
    x = energy_right - energy_left
    y = energy_front - energy_back
    angle = np.degrees(np.arctan2(y, x)) % 360 #compute angle and convert to degrees

    #calculate intensity & normalize to 0-1
    magnitude = np.sqrt(x*x + y*y)
    intensity = magnitude / (magnitude + 1e-6)
    
    return {
    "angle": angle,
    "intensity": intensity,
    "raw_energies": {
        "front": energy_front,
        "back": energy_back,
        "left": energy_left,
        "right": energy_right,
        "channels": energies
    }
}