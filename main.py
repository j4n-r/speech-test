import time
import wave
from faster_whisper import WhisperModel
from piper import PiperVoice

def main():
    # 1. Transcribe
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    # Transcribe
    segments, info = model.transcribe("test.wav", beam_size=5, language="de")

    print(f"Detected language: {info.language} ({info.language_probability:.2f})")

    text_segments = []
    for segment in segments:
        text_segments.append(segment.text)

    full_text = " ".join(text_segments).strip()
    print("Transcribed text:", full_text)
    return full_text

def text_to_speech(text):
    # 2. Synthesize (TTS)
    voice = PiperVoice.load("de_DE-eva_k-x_low.onnx",
                            config_path="de_DE-eva_k-x_low.json")

    # Open a NEW file for output
    output_filename = "output.wav"
    
    with wave.open(output_filename, "wb") as wav_file:
        # Use synthesize_wav if available (seems to be working in your version)
        voice.synthesize_wav(text, wav_file)

    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    now = time.time()
    
    # Run STT
    text = main()
    
    then = time.time()
    print("Time speech to text:", then - now)
    
    if text:
        tts_now = time.time()
        text_to_speech(text)
        tts_then = time.time()
        print("Time text to speech:", tts_then - tts_now)
        print("Total time:", tts_then - now)
    else:
        print("No text detected.")
