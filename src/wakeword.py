import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel

# --- Configuration ---
RATE = 16000
CHUNK = 1280
WAKE_WORD = "hey_jarvis"

# Silence Detection Settings
THRESHOLD = 500       # Volume threshold (0-32768). Adjust if mic is sensitive.
SILENCE_LIMIT = 2     # Seconds of silence before stopping
MAX_TIME = 10         # Safety limit: stop recording after 10s no matter what

def main():
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

            if ww_model.predict(audio_int16)[WAKE_WORD] > 0.5:
                print(f"\n[!] Wake word detected! Listening...")

                # --- Active Phase: Record until Silence ---
                frames = []
                silence_chunks = 0
                max_chunks = int(RATE / CHUNK * MAX_TIME)
                chunks_for_silence = int(RATE / CHUNK * SILENCE_LIMIT)

                for _ in range(max_chunks):
                    data = stream.read(CHUNK)
                    chunk = np.frombuffer(data, dtype=np.int16)
                    frames.append(chunk)

                    # Calculate "Loudness" (Mean Absolute Amplitude)
                    loudness = np.abs(chunk).mean()

                    # Check if silence
                    if loudness < THRESHOLD:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0 # Reset if we hear noise

                    # Stop if we have enough silence chunks in a row
                    if silence_chunks > chunks_for_silence:
                        print("[-] Silence detected. Stopping.")
                        break

                # --- Processing Phase ---
                print("Transcribing...")
                audio_float = np.concatenate(frames).flatten().astype(np.float32) / 32768.0
                
                segments, _ = whisper.transcribe(audio_float, beam_size=5)
                text = " ".join([s.text for s in segments]).strip()
                
                print(f"[>] You said: '{text}'\n")

                ww_model.reset()
                print("Waiting...")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.close()
        mic.terminate()

if __name__ == "__main__":
    main()
