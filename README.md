# RoboDino

RoboDino is an embodied interactive robot project (based on a quadruped robot) designed to be a cute, reactive, lifelike creature rather than just a passive voice assistant. 

This repository contains the high-level cognitive "brain" of RoboDino, which handles multimodal perception and intents.

## Current MVP Architecture (100% Free & Local)

RoboDino's current interaction pipeline translates human speech into structured physical behaviors without relying on paid cloud APIs.

1. **Audio Listener (`audio_listener.py`)**: Uses microphone input and `webrtcvad` (Voice Activity Detection) to intelligently listen bridging silence gaps, saving segmented user speech.
2. **ASR Module (`intent_engine.py`)**: Uses the local `faster-whisper` model to transcribe the recorded audio into text quickly and locally.
3. **Intent Parser (`intent_engine.py`)**: Sends the transcribed text to a local Large Language Model (Qwen2.5:1.5b running via Ollama) which is strictly prompted to output a JSON object representing the robot's intended physical reaction (e.g., `{"behavior": "excited"}`).
4. **Main Loop (`main_loop.py`)**: The orchestrator that continuously bridges audio capture, transcription, and intent parsing.

## Available Behaviors
The LLM is constrained to output only the following safe, predefined behaviors:
* `greet` - Friendly acknowledgment or waving
* `curious` - Tilting head, looking closer
* `excited` - Happy taps, wagging
* `present` - Showing off something
* `goodbye` - Turning away or waving bye
* `idle` - Default ambient state (breathing, slight swaying)
* `confused` - When input is unclear or intent is unknown

*(Note: These high-level intents must be mapped to specific motor/SDK commands in your robot's lower-level control software like MuJoCo or the Unitree Go2 SDK).*

---

## Environment Setup

### 1. Prerequisites
You need a system with a microphone and ideally an NVIDIA GPU for fast local LLM inference.

Install system-level audio dependencies:
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

### 2. Install Ollama & Qwen Model
Ollama is required to run the local LLM intent parser for free.
```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start the service (if not auto-started)
# sudo systemctl start ollama

# 3. Pull the Qwen model
ollama run qwen2.5:1.5b
# (Type /bye to exit the chat prompt once downloaded)
```

### 3. Python Environment
Create a Conda environment and install dependencies:
```bash
conda create -n RobotDino python=3.10 -y
conda activate RobotDino
pip install faster-whisper sounddevice numpy scipy webrtcvad requests
```

---

## Usage

**To run the full voice-to-intent interactive loop:**
```bash
conda activate RobotDino
python main_loop.py
```
* Speak clearly into your computer's microphone.
* The system will automatically detect when you stop speaking, process the audio, and output the resulting JSON behavior intent to the console.
* Say "exit" or "quit" to stop the loop.

**To run a text-only test (No Microphone):**
```bash
python test_pipeline.py
```

## Legacy Files
* `Voice.py`: The original proof-of-concept testing script that relied on the paid OpenAI Realtime WebSocket API. It remains for reference.
# ROBOTDino
