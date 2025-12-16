# ELIZA Voice

A Python implementation of ELIZA, the classic 1966 chatbot, with voice interaction capabilities. Talk to the "therapist" using speech-to-text and text-to-speech.

Based on Charles Hayden's Java version at http://chayden.net/eliza/Eliza.html.

## Features

- Classic ELIZA chatbot behavior
- Voice interaction mode using:
  - **Speech-to-Text**: OpenAI Whisper (runs locally)
  - **Text-to-Speech**: Piper TTS (fast, local neural TTS)
- Works entirely offline after initial model download

## Installation

### Prerequisites

- Python 3.9+
- ffmpeg (for audio processing)

On macOS:
```bash
brew install ffmpeg
```

### Setup

```bash
# Clone the repository
git clone https://github.com/RogueSergeant/eliza-voice.git
cd eliza-voice

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Text Mode (Default)

```bash
python eliza.py
```

```
How do you do.  Please tell me your problem.
> I would like to have a chat bot.
You say you would like to have a chat bot ?
> bye
Goodbye.  Thank you for talking to me.
```

### Voice Mode

```bash
python eliza.py --voice
```

In voice mode:
1. ELIZA speaks its greeting
2. Speak your response (recording stops after 1.5s of silence)
3. ELIZA responds with speech
4. Say "bye", "goodbye", or "quit" to end the session

### As a Library

```python
import eliza

eliza = eliza.Eliza()
eliza.load('doctor.txt')

print(eliza.initial())
while True:
    said = input('> ')
    response = eliza.respond(said)
    if response is None:
        break
    print(response)
print(eliza.final())
```

## Testing

```bash
# Run all tests
python -m unittest discover

# Run specific test file
python -m unittest test_voice
python -m unittest test_eliza
```

## Models

On first run with `--voice`, the following models will be downloaded:
- **Whisper "base"** (~140MB) - for speech recognition
- **Piper "en_US-lessac-medium"** - for text-to-speech

Models are cached locally for subsequent runs.