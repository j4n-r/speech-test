import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
WAKE_WORD_THRESHOLD = 0.5
COMMAND_DURATION = 5  # Seconds to record after wake word

def load_models() -> tuple[Model, WhisperModel]:
    """Downloads and loads the Wake Word and Whisper models."""
    print("[*] Loading models...")
    openwakeword.utils.download_models()
    
    # Load Wake Word Model
    ww_model = Model(wakeword_models=["alexa", "hey_jarvis"])
    
    # Load Whisper Model
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    
    return ww_model, whisper_model

def get_audio_stream(p: pyaudio.PyAudio) -> pyaudio.Stream:
    """Configures and opens the microphone stream."""
    return p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

def capture_and_transcribe(stream: pyaudio.Stream, whisper: WhisperModel) -> str:
    """Records audio for a set duration and returns the transcribed text."""
    print("[*] Listening for command... ['alexa' or 'hey jarvis']")
    frames: list[np.ndarray] = []

    # Record audio
    for _ in range(0, int(RATE / CHUNK * COMMAND_DURATION)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("[*] Processing...")
    
    # Convert buffer to float32 for Whisper (normalization)
    audio_data = np.concatenate(frames).flatten().astype(np.float32) / 32768.0

    # Transcribe
    segments, _ = whisper.transcribe(audio_data, beam_size=5)
    text = "".join([s.text for s in segments]).strip()

    return text

def main():
    # 1. Setup Resources
    ww_model, whisper_model = load_models()
    audio = pyaudio.PyAudio()
    stream = get_audio_stream(audio)

    print("[*] System Ready. Waiting for wake word...")

    try:
        while True:
            # Get audio chunk
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16)

            # Predict Wake Word
            prediction = ww_model.predict(audio_frame)

            # Check for triggers
            for mdl_name, score in prediction.items():
                if score > WAKE_WORD_THRESHOLD:
                    print(f"\n[!] Wake Word Detected: {mdl_name}")
                    
                    # Capture Command
                    command_text = capture_and_transcribe(stream, whisper_model)
                    
                    if command_text:
                        print(f"[>] User said: '{command_text}'\n")
                    else:
                        print("[>] No speech detected.\n")

                    # Reset model buffer to prevent immediate re-trigger
                    ww_model.reset()
                    print("[*] Waiting for wake word...")

    except KeyboardInterrupt:
        print("\n[*] Stopping...")

    finally:
        # Cleanup
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
