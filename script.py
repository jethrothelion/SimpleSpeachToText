#we get file name first before load everything else as not to have
#the user wait after hitting run
import argparse
import os
from datetime import datetime

# --- command line arg set up ---
parser = argparse.ArgumentParser(description="Real-time audio transcription to a text file.")
subparsers = parser.add_subparsers(dest='command', help='Available commands')

#  'new' command: Creates a new file
parser_new = subparsers.add_parser('new', help='Create a new transcription file.')
parser_new.add_argument('name', type=str, help='The base name for the new file (e.g., "Meeting Notes").')

# 'append' command: Appends to an existing file
parser_append = subparsers.add_parser('append', help='Append to an existing transcription file.')
parser_append.add_argument('file', type=argparse.FileType('a'), help='The full path to the file to append to.')

args = parser.parse_args()


def get_filename():
    while True:
        Filenamemode = input("Append or new file? (A or N) ").lower().strip()
        if Filenamemode == "a":
            filename = input("Enter filename: ")
            if os.path.exists(filename):
                return filename
            else:
                print("no existing file, please enter another name")
                continue

        elif Filenamemode == "n":
            classname = input("Enter desired file name: ")
            filename = f"{classname} {timestamp}.txt"
            return filename

        else:
            print("Invalid choice, please enter 'A' or 'N'.")

timestamp = datetime.now().strftime("%Y-%m-%d")

if args.command is None:
    filename = get_filename()
elif args.command == 'new':
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{args.name}{timestamp}.txt"
elif args.command  =='append':
    filename = args.file.name
    args.file.write(f"\n--- Appending on {timestamp} ---\n")
    args.file.close()

import queue
import sys
import threading
import sounddevice as sd
import numpy as np
import nemo.collections.asr as nemo_asr
import configparser
import webrtcvad

# --- read config variables ---
config = configparser.ConfigParser()
config.read('config.ini')

model_name = config.get('model', 'model_name')
samplerate = config.getint('audio', 'samplerate')
chunk_seconds = config.getint('audio', 'chunk_seconds')
vad_state = config.getboolean('vad', 'Voice_activation')
vad_Aggressiveness = config.getint('vad', 'Aggressiveness')
chunk_size = chunk_seconds * samplerate

# --- VAD Setup ---
#if vad_state == True: ##One day do this, still using samples perframe from here
vad = webrtcvad.Vad()
vad.set_mode(vad_Aggressiveness)  # Set aggressiveness mode from 0 (least aggressive) to 3 (most aggressive)
VAD_FRAME_DURATION_MS = 30
VAD_SAMPLES_PER_FRAME = int(samplerate * VAD_FRAME_DURATION_MS / 1000)


# --- Events and queues ---
pause_event = threading.Event()
quit_event = threading.Event()
pause_event.set()
mic_q = queue.Queue()
transcription_q = queue.Queue()

# --- Thread-safe cb_status bar ---
print_lock = threading.Lock()
counts = {"mic": 0, "backlog": 0}


def status(msg,color="\x1b[0m"): #default color if no color passed
    """prints message on seperate cb_status bar on the bottom line"""
    with print_lock:
        if pause_event.is_set():
            state = "LIVE"
        else:
            state = "PAUSED"
        clear_colors = "\x1b[0m"
        state_color = "\x1b[95m"
        input_color = "\x1b[96m"
        sys.stdout.write(f"\r\033{color}{msg}\n  {state_color}[{state}] Mic: {counts['mic']} | Backlog: {counts['backlog']}\n{input_color}p, q, r or input text>>{clear_colors}")
        sys.stdout.flush()

def refresh_status():
    """redraws the cb_status bar in place."""
    with print_lock:
        if pause_event.is_set():
            state = "LIVE"
        else:
            state = "PAUSED"
        sys.stdout.write(f"\033[s\033[1A\r\033[{state}] Mic: {counts['mic']} | Backlog: {counts['backlog']}\033[u")
        sys.stdout.flush()



# --- Load ASR model ---
model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
print("Model loaded")



def callback(indata, frames, time, cb_status):
    if not pause_event.is_set():  # If paused, do nothing
        return
    if cb_status:
        print(f"Audio Status: {cb_status}\n>> ", end="")
    # Always flatten to 1-D
    mic_q.put(indata.flatten().astype(np.float32))


def console_input_thread():
    while True:
        user_text = input(" ").lower().strip()
        if user_text in ("pause", 'p'):
            status("\n--- Transcription PAUSED. Type 'resume' to continue. ---\n", "\x1b[31m")
            pause_event.clear()
        elif user_text in ("resume", "r", "go"):
            status("\n--- Transcription RESUMED. ---\n", "\x1b[32m")
            pause_event.set()
        elif user_text in ("exit", "quit", "q"):
            status("\n--- Transcription EXIT ATTEMPTING. ---\n", "\x1b[91m")
            pause_event.clear()
            quit_event.set()
            break
        else:
            with open(filename, "a", encoding="utf-8") as f:
                f.write("[USER]: " + user_text + "\n")
            status(f"Added to file. {user_text}")

def transcription():
    while not quit_event.is_set():
        try:
            # Wait for up to 0.1s for a chunk to be ready to transcribe
            audio_to_process = transcription_q.get(timeout=0.1)

            result = model.transcribe(audio=[audio_to_process], batch_size=1)



            transcribed_text = result[0].text if result and result[0].text else " "
            status(f"Transcribed: {transcribed_text}", "\x1b[93m") #yellow
            if transcribed_text:
                try:
                    with open(filename, "a", encoding="utf-8") as f:
                        f.write("[ASR]:" + transcribed_text + "\n")
                except Exception as e:
                    print(f"file writing error while {e}")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Transcription Error: {e}")


threading.Thread(target=console_input_thread, daemon=True).start()
threading.Thread(target=transcription, daemon=True).start()

print("Streaming from microphone... Press Ctrl+C to stop.")

try:
    transcription_buffer = []
    processing_buffer = np.array([], dtype=np.float32)
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        while not quit_event.is_set() or not transcription_q.empty():
            try:
                audio_chunk = mic_q.get(timeout=0.1)


                processing_buffer = np.concatenate([processing_buffer, audio_chunk.flatten()])
            except queue.Empty:
                continue

            counts["mic"] = mic_q.qsize()
            counts["backlog"] = transcription_q.qsize()


            while len(processing_buffer) >= VAD_SAMPLES_PER_FRAME:

                frame = processing_buffer[:VAD_SAMPLES_PER_FRAME]
                processing_buffer = processing_buffer[VAD_SAMPLES_PER_FRAME:]

                #If volume silent skip loop
                frame_volume = np.linalg.norm(frame)
                if frame_volume < 1:
                    continue

                if not vad_state:
                    transcription_buffer.append(frame)
                else:
                    # --- Convert audio for VAD ---
                    # 1. Convert from float32 to int16
                    frame_int16 = (frame * 32767).astype(np.int16)
                    # 2. Convert from int16 to bytes
                    frame_bytes = frame_int16.tobytes()
                    is_speech = vad.is_speech(frame_bytes, samplerate)
                    if is_speech:
                        transcription_buffer.append(frame)

            if transcription_buffer:
                audio_float = np.concatenate(transcription_buffer, axis=0)

                if len(audio_float) >= chunk_size:
                    to_process = audio_float[:chunk_size]

                    remaining_audio = audio_float[chunk_size:]
                    transcription_buffer = [remaining_audio] if len(remaining_audio) > 0 else []

                    transcription_q.put(to_process)
        os._exit(0)


except KeyboardInterrupt:
    print("\nStopped by user.")
    os._exit(0)
except Exception as e:
    print(f"Error: {e}")

