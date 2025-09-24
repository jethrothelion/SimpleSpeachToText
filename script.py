import sounddevice as sd
import numpy as np
import queue
import threading
import nemo.collections.asr as nemo_asr
from datetime import datetime
from absl.logging import exception

pause_event = threading.Event()
pause_event.set()

timestamp = datetime.now().strftime("%Y-%m-%d")


while True:
    Filenamemode = input("Append or new file? (A or N) ")
    if Filenamemode.lower() == "a":
        filename = input("Enter filename: ")
        with open(filename, "a", encoding="utf-8") as f:
            f.write(timestamp + "\n")
        break
    elif Filenamemode.lower() == "n":
        classname = input("Enter class name: ")
        filename = f"{classname} {timestamp}.txt"
        break
    else:
        print("Invalid choice, please enter 'A' or 'N'.")



# --- Load ASR model ---
model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
print("Model loaded")

# --- Audio streaming setup ---
samplerate = 16000
chunk_size = 5 * samplerate  # 5 seconds worth of audio
q = queue.Queue()

def callback(indata, frames, time, status):
    if not pause_event.is_set(): # If paused, do nothing
        return
    if status:
        print(status, flush=True)
    # Always flatten to 1-D
    q.put(indata.flatten().astype(np.float32))

def console_input_thread():
    while True:
        user_text = input().lower()
        with open(filename, "a", encoding="utf-8") as f:
            f.write("[USER]: " + user_text + "\n")
        print("Added to file.")
        if user_text == "pause":
            print("\n--- Transcription PAUSED. Type 'resume' to continue. ---\n")
            pause_event.clear() #
        elif user_text == "resume":
            print("\n--- Transcription RESUMED. ---\n")
            pause_event.set()


threading.Thread(target=console_input_thread, daemon=True).start()

print("Streaming from microphone... Press Ctrl+C to stop.")

try:
    buffer = []
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        while True:
            try:
                audio_chunk = q.get(timeout=0.1)
            except queue.Empty:
                continue
            buffer.append(audio_chunk)  # always 1-D now

            audio_float = np.concatenate(buffer, axis=0)

            # Only process when we have >= 5 seconds
            if len(audio_float) >= chunk_size:
                to_process = audio_float[:chunk_size]

                # keep leftover
                buffer = [audio_float[chunk_size:]]

                print("Transcribing...")
                result = model.transcribe(audio=[to_process], batch_size=1)
                print("Transcribed:", result[0].text)

                with open(filename, "a", encoding="utf-8") as f:
                    f.write("[ASR]:" + result[0].text + "\n")


except KeyboardInterrupt:
    print("\nStopped by user.")
except Exception as e:
    print(f"Error: {e}")
