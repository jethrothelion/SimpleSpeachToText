import sounddevice as sd
import numpy as np
import queue
import sys
import threading
import nemo.collections.asr as nemo_asr
from datetime import datetime
from absl.logging import exception
import configparser
import webrtcvad

# buncha bs
config = configparser.ConfigParser()
config.read('config.ini')

# get config variables
model_name = config.get('model', 'model_name')
samplerate = config.getint('audio', 'samplerate')
chunk_seconds = config.getint('audio', 'chunk_seconds')
vad_state = config.getboolean('vad', 'Voice_activation')
vad_Aggressiveness = config.getboolean('vad', 'Aggressiveness')
chunk_size = chunk_seconds * samplerate

pause_event = threading.Event()
quit_event = threading.Event()
pause_event.set()

timestamp = datetime.now().strftime("%Y-%m-%d")

# --- VAD Setup ---
vad = webrtcvad.Vad()
vad.set_mode(vad_Aggressiveness)  # Set aggressiveness mode from 0 (least aggressive) to 3 (most aggressive)
VAD_FRAME_DURATION_MS = 30
VAD_SAMPLES_PER_FRAME = int(samplerate * VAD_FRAME_DURATION_MS / 1000)
frame_duration = 30
samples_per_frame = int(samplerate * frame_duration / 1000)

while True:
    Filenamemode = input("Append or new file? (A or N) ")
    if Filenamemode.lower() == "a":
        filename = input("Enter filename: ")
        with open(filename, "a", encoding="utf-8") as f:
            f.write(timestamp + "\n")
        break
    elif Filenamemode.lower() == "n":
        classname = input("Enter desired file name: ")
        filename = f"{classname} {timestamp}.txt"
        break
    else:
        print("Invalid choice, please enter 'A' or 'N'.")

# --- Load ASR model ---
model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
print("Model loaded")

# --- Audio streaming setup ---
q = queue.Queue()


def callback(indata, frames, time, status):
    if not pause_event.is_set():  # If paused, do nothing
        return
    if status:
        print(f"Audio Status: {status}\n>> ", end="")
    # Always flatten to 1-D
    q.put(indata.flatten().astype(np.float32))


def console_input_thread():
    while True:
        user_text = input("\nEnter command (pause, exit) or put text in file: \n").lower()
        with open(filename, "a", encoding="utf-8") as f:
            f.write("[USER]: " + user_text + "\n")
        print("Added to file.")
        if user_text in ("pause", 'p'):
            print("\n--- Transcription PAUSED. Type 'resume' to continue. ---\n")
            pause_event.clear()  #
        elif user_text in ("resume", "r", "go"):
            print("\n--- Transcription RESUMED. ---\n")
            pause_event.set()
        elif user_text in ("exit", "quit", "q"):
            print("\n--- Transcription EXIT ATTEMPTING. ---\n")
            pause_event.clear()
            quit_event.set()
            break


threading.Thread(target=console_input_thread, daemon=True).start()

print("Streaming from microphone... Press Ctrl+C to stop.")

try:
    transcription_buffer = []
    processing_buffer = np.array([], dtype=np.float32)  # This holds RAW audio from the mic
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        while not quit_event.is_set():
            try:
                audio_chunk = q.get(timeout=0.1)

                processing_buffer = np.concatenate([processing_buffer, audio_chunk.flatten()])
            except queue.Empty:
                continue
            while len(processing_buffer) >= VAD_SAMPLES_PER_FRAME:

                frame_to_process = processing_buffer[:VAD_SAMPLES_PER_FRAME]
                processing_buffer = processing_buffer[VAD_SAMPLES_PER_FRAME:]
                if not vad_state:
                    transcription_buffer.append(frame_to_process)
                else:
                    # --- Convert audio for VAD ---
                    # 1. Convert from float32 to int16
                    frame_int16 = (frame_to_process * 32767).astype(np.int16)
                    # 2. Convert from int16 to bytes
                    frame_bytes = frame_int16.tobytes()
                    is_speech = vad.is_speech(frame_bytes, samplerate)
                    if is_speech:
                        transcription_buffer.append(frame_to_process)
                        print("speach")

                if transcription_buffer:
                    audio_float = np.concatenate(transcription_buffer, axis=0)

                    if len(audio_float) >= chunk_size:
                        to_process = audio_float[:chunk_size]

                        remaining_audio = audio_float[chunk_size:]
                        transcription_buffer = [remaining_audio] if len(remaining_audio) > 0 else []

                        print("Transcribing")
                        result = model.transcribe(audio=[to_process], batch_size=1)
                        transcribed_text = result[0].text if result and result[0].text else None

                        print("Transcribed:", result[0].text)
                        if transcribed_text:
                            with open(filename, "a", encoding="utf-8") as f:
                                f.write("[ASR]:" + result[0].text + "\n")
        sys.exit()


except KeyboardInterrupt:
    print("\nStopped by user.")
except Exception as e:
    print(f"Error: {e}")

