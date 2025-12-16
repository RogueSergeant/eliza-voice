# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python implementation of ELIZA, the classic 1966 chatbot created by Joseph Weizenbaum, with voice interaction capabilities. It simulates a Rogerian psychotherapist using pattern matching and substitution, and supports both text and voice modes.

## Commands

**Setup (first time):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run in text mode:**
```bash
python eliza.py
```

**Run in voice mode:**
```bash
python eliza.py --voice
```

**Run all tests:**
```bash
python -m unittest discover
```

**Run specific test file:**
```bash
python -m unittest test_eliza
python -m unittest test_voice
```

**Run a single test:**
```bash
python -m unittest test_eliza.ElizaTest.test_response_1
```

## Architecture

The system has four main components:

1. **Script Parser** (`Eliza.load()`) - Parses `doctor.txt` which defines:
   - `initial`/`final`: Greeting and goodbye messages
   - `quit`: Words that end the conversation (bye, goodbye, quit)
   - `pre`/`post`: Word substitutions applied before/after pattern matching
   - `synon`: Synonym groups referenced with `@` in patterns (e.g., `@sad` matches unhappy, depressed, sick)
   - `key`: Keywords with weights (higher = priority) containing decomposition rules
   - `decomp`: Patterns using `*` (wildcard) and `@group` (synonym match)
   - `reasmb`: Response templates with `(n)` placeholders or `goto` redirects

2. **Pattern Matcher** (`_match_decomp`, `_match_decomp_r`) - Recursive matching of decomposition patterns against tokenized input. Returns captured groups for reassembly.

3. **Response Generator** (`respond`, `_match_key`, `_reassemble`) - Processes input through pre-substitutions, matches against weighted keywords, applies post-substitutions to captured groups, and constructs responses. Maintains a memory queue for deferred responses marked with `$`.

4. **Voice Interface** (`voice.py`) - Handles speech I/O:
   - `VoiceInterface.record_audio()` - Records from microphone with silence detection
   - `VoiceInterface.transcribe()` - Converts speech to text using Whisper
   - `VoiceInterface.speak()` - Converts text to speech using Piper TTS
   - Models are lazy-loaded and cached locally

## Script Format (doctor.txt)

The script uses a simple `tag: content` format. Keywords have optional weights (`key: remember 5`). Decomposition patterns under a key use indentation. The special `xnone` key provides fallback responses when no patterns match.
