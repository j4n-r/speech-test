import pyaudio
import numpy as np

# Same settings as the main script
RATE = 16000
CHUNK = 1280

def main():
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n[*] Monitoring Microphone Volume...")
    print("[*] PRESS CTRL+C TO STOP\n")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Convert to numbers
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate volume
            volume = int(np.abs(audio_data).mean())
            
            # Create a visual bar (divide by 100 to make it fit on screen)
            bar = "|" * int(volume / 50) 
            
            # Print the number and the bar
            print(f"Volume: {volume:4d} {bar}")

    except KeyboardInterrupt:
        print("\nStopping...")
        stream.close()
        mic.terminate()

if __name__ == "__main__":
    main()
