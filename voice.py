"""Voice module for ELIZA - provides speech-to-text and text-to-speech capabilities."""

import io
import logging
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

# Audio recording settings
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1
DTYPE = np.int16
SILENCE_THRESHOLD = 500  # RMS threshold for silence detection
SILENCE_DURATION = 1.5  # Seconds of silence to stop recording
MAX_RECORDING_DURATION = 30  # Maximum recording length in seconds


class VoiceInterface:
    """Handles speech-to-text and text-to-speech for ELIZA."""

    def __init__(self, whisper_model: str = "base", piper_voice: str = "en_US-lessac-medium"):
        """
        Initialize the voice interface.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            piper_voice: Piper voice model name
        """
        self.whisper_model_name = whisper_model
        self.piper_voice_name = piper_voice
        self._whisper_model = None
        self._piper_voice = None

    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self._whisper_model is None:
            import whisper
            log.info(f"Loading Whisper model: {self.whisper_model_name}")
            self._whisper_model = whisper.load_model(self.whisper_model_name)
        return self._whisper_model

    def _load_piper(self):
        """Lazy load Piper voice."""
        if self._piper_voice is None:
            from piper import PiperVoice
            log.info(f"Loading Piper voice: {self.piper_voice_name}")
            # Piper will download the model if not present
            model_path = self._get_piper_model_path()
            self._piper_voice = PiperVoice.load(str(model_path))
        return self._piper_voice

    def _get_piper_model_path(self) -> Path:
        """Get or download the Piper model."""
        from piper.download_voices import download_voice

        # Check standard piper cache location
        cache_dir = Path.home() / ".local" / "share" / "piper_tts"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = cache_dir / f"{self.piper_voice_name}.onnx"

        if not model_path.exists():
            log.info(f"Downloading Piper voice model: {self.piper_voice_name}")
            download_voice(self.piper_voice_name, cache_dir)

        return model_path

    def record_audio(self) -> np.ndarray:
        """
        Record audio from microphone until silence is detected.

        Returns:
            Audio data as numpy array
        """
        log.debug("Starting audio recording...")
        print("Listening... (speak now)")

        frames = []
        silence_frames = 0
        frames_per_check = int(SAMPLE_RATE * 0.1)  # Check every 100ms
        silence_frames_needed = int(SILENCE_DURATION / 0.1)
        max_frames = int(MAX_RECORDING_DURATION / 0.1)

        def callback(indata, frame_count, time_info, status):
            if status:
                log.warning(f"Recording status: {status}")
            frames.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype=DTYPE, blocksize=frames_per_check,
                            callback=callback):
            frame_count = 0
            while frame_count < max_frames:
                sd.sleep(100)  # Sleep 100ms
                frame_count += 1

                if len(frames) > 0:
                    # Check if recent audio is silence
                    recent = frames[-1]
                    rms = np.sqrt(np.mean(recent.astype(np.float32) ** 2))

                    if rms < SILENCE_THRESHOLD:
                        silence_frames += 1
                        if silence_frames >= silence_frames_needed and len(frames) > silence_frames_needed:
                            log.debug(f"Silence detected after {len(frames)} frames")
                            break
                    else:
                        silence_frames = 0

        if not frames:
            return np.array([], dtype=DTYPE)

        audio_data = np.concatenate(frames, axis=0).flatten()
        log.debug(f"Recorded {len(audio_data)} samples ({len(audio_data)/SAMPLE_RATE:.1f}s)")
        return audio_data

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio_data: Audio data as numpy array (int16)

        Returns:
            Transcribed text
        """
        if len(audio_data) == 0:
            return ""

        model = self._load_whisper()

        # Convert to float32 normalized to [-1, 1] as Whisper expects
        audio_float = audio_data.astype(np.float32) / 32768.0

        log.debug("Transcribing audio...")
        result = model.transcribe(audio_float, language="en")
        text = result["text"].strip()
        log.debug(f"Transcribed: {text}")
        return text

    def speak(self, text: str):
        """
        Convert text to speech and play it.

        Args:
            text: Text to speak
        """
        if not text:
            return

        log.debug(f"Speaking: {text}")

        # Use piper CLI for simplicity and reliability
        import subprocess
        import sys

        # Get the full path to the model
        model_path = self._get_piper_model_path()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate audio with piper using full model path
            result = subprocess.run(
                [sys.executable, "-m", "piper", "--model", str(model_path),
                 "--output_file", tmp_path],
                input=text.encode(),
                capture_output=True
            )

            if result.returncode != 0:
                log.error(f"Piper error: {result.stderr.decode()}")
                print(text)  # Fallback to printing
                return

            # Play the audio
            self._play_wav(tmp_path)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _play_wav(self, wav_path: str):
        """Play a WAV file using sounddevice."""
        with wave.open(wav_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        sd.play(audio_data, sample_rate)
        sd.wait()

    def listen_and_transcribe(self) -> str:
        """
        Record audio and transcribe it to text.

        Returns:
            Transcribed text from user speech
        """
        audio = self.record_audio()
        return self.transcribe(audio)


def test_voice():
    """Quick test of voice functionality."""
    logging.basicConfig(level=logging.DEBUG)

    voice = VoiceInterface()

    print("Testing TTS...")
    voice.speak("Hello, I am ELIZA. How do you do?")

    print("\nTesting STT - please say something...")
    text = voice.listen_and_transcribe()
    print(f"You said: {text}")


if __name__ == "__main__":
    test_voice()
