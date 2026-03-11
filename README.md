# 🗣️ SimpleSpeechToText

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

A simple voice to text Python transcription script focusing on the balance between power draw, simplicity and accuracy using the nvidia/parakeet-tdt-0.6b-v2 model

Very handy for Meetings and Lectures on a laptop or a pi, Saves to file and outputs to terminal

---

## ✨ Features

* **Offline Conversion**: Works entirely offline, no internet connection needed other then to download the model
* **Playback Control**: Pause and resume the Transcription at any time.
* **Cross-Platform**: Tested with Linux and windows 10
* **Lightweight**: Minimal dependencies, a small footprint and easy to run
* **User Input** You can add text along side the transcription with a different line header to differentiate
* **Append or new file** Asks if want to create a new or add on to a existing file with command line args available

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine. Assuming your microphone is already set up

### Prerequisites

I used python 12 but versions within reason might work

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/jethrothelion/SimpleTextToSpeech.git](https://github.com/jethrothelion/SimpleTextToSpeech.git)
    cd SimpleTextToSpeech
    ```

2.  **Install the required packages:**
    The project relies on the `Nemo`, `Numpy` and other libraries. Install them using pip:
    ```sh
    pip install -r requirements.txt
    ```
---

## 💻 Usage

Using the `SimpleTextToSpeech` script is straightforward. Just run the python file and it will ask all it needs to know

Feel free to build off of this very simple file


1.  **Default behavior with no argumants:**
    ```sh
    python3 script.py
    ```
it will ask for new or append to file and then go

2.  **Specify new file (will create a new file in the code directory):**
    ```sh
    python3 script.py new "Meating notes"
    ```

2.  **Specify append a file:**
    ```sh
    python3 script.py append "C:/Full/path/to/existing/file"
    ```

