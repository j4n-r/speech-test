import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel

# --- Configuration ---
RATE = 16000          # Standard sample rate for these models
CHUNK = 1280          # Audio chunk size (1280 samples = 80ms)
WAKE_WORD = "hey_jarvis"
LISTEN_TIME = 5       # Seconds to record after wake word

def main():
    # 1. Setup Resources
    print("Loading models...")
    openwakeword.utils.download_models()
    ww_model = Model(wakeword_models=[WAKE_WORD])
    whisper = WhisperModel("small", device="cpu", compute_type="int8")

    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print(f"System Ready. Waiting for '{WAKE_WORD}'...")

    try:
        while True:
            # --- Passive Phase: Listen for Wake Word ---
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_int16 = np.frombuffer(data, dtype=np.int16)

            # Get prediction (returns a dict like {'hey_jarvis': 0.002})
            prediction = ww_model.predict(audio_int16)

            # Check if confidence is above 50%
            if prediction[WAKE_WORD] > 0.5:
                print(f"\n[!] Wake word detected! Recording for {LISTEN_TIME}s...")

                # --- Active Phase: Record Command ---
                frames = []
                # Calculate how many chunks we need to read to match LISTEN_TIME
                chunks_to_record = int(RATE / CHUNK * LISTEN_TIME)
                
                for _ in range(chunks_to_record):
                    data = stream.read(CHUNK)
                    frames.append(np.frombuffer(data, dtype=np.int16))

                # --- Processing Phase: Transcribe ---
                print("Transcribing...")
                
                # Combine frames and normalize audio to float32 (required by Whisper)
                audio_float = np.concatenate(frames).flatten().astype(np.float32) / 32768.0
                
                segments, _ = whisper.transcribe(audio_float, beam_size=5)
                text = " ".join([s.text for s in segments]).strip()
                
                print(f"[>] User said: '{text}'\n")

                # Reset wake word buffer so it doesn't trigger on its own echo
                ww_model.reset()
                print("Waiting...")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.close()
        mic.terminate()

if __name__ == "__main__":
    main()
