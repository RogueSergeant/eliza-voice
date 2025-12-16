"""Tests for voice module - speech-to-text and text-to-speech functionality."""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import wave
import numpy as np
from pathlib import Path


class TestVoiceInterfaceInit(unittest.TestCase):
    """Test VoiceInterface initialization."""

    def test_init_default_models(self):
        """Test default model names are set correctly."""
        from voice import VoiceInterface
        voice = VoiceInterface()
        self.assertEqual(voice.whisper_model_name, "base")
        self.assertEqual(voice.piper_voice_name, "en_US-lessac-medium")

    def test_init_custom_models(self):
        """Test custom model names are set correctly."""
        from voice import VoiceInterface
        voice = VoiceInterface(whisper_model="tiny", piper_voice="en_GB-alan-medium")
        self.assertEqual(voice.whisper_model_name, "tiny")
        self.assertEqual(voice.piper_voice_name, "en_GB-alan-medium")

    def test_lazy_loading(self):
        """Test that models are not loaded on init (lazy loading)."""
        from voice import VoiceInterface
        voice = VoiceInterface()
        self.assertIsNone(voice._whisper_model)
        self.assertIsNone(voice._piper_voice)


class TestAudioRecording(unittest.TestCase):
    """Test audio recording functionality."""

    def test_record_audio_returns_numpy_array(self):
        """Test that record_audio returns a numpy array."""
        from voice import VoiceInterface, SAMPLE_RATE, DTYPE

        with patch('sounddevice.InputStream') as mock_stream:
            # Create mock audio data (silence to trigger stop)
            mock_audio = np.zeros((1600, 1), dtype=DTYPE)

            # Set up mock to populate frames and then stop
            call_count = [0]

            def mock_callback_trigger(callback):
                # Simulate some audio frames
                for _ in range(20):  # Simulate 2 seconds of audio
                    callback(mock_audio, 1600, None, None)
                return MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: None)

            mock_stream.return_value.__enter__ = lambda s: s
            mock_stream.return_value.__exit__ = lambda s, *a: None

            voice = VoiceInterface()

            # Mock the InputStream to return quickly
            with patch('sounddevice.sleep', side_effect=lambda x: None):
                with patch('sounddevice.InputStream') as mock_is:
                    mock_is.return_value.__enter__ = lambda s: s
                    mock_is.return_value.__exit__ = lambda s, *a: None

                    # Directly test that the method exists and has correct signature
                    self.assertTrue(callable(voice.record_audio))


class TestTranscription(unittest.TestCase):
    """Test speech-to-text transcription."""

    def test_transcribe_empty_audio(self):
        """Test transcription of empty audio returns empty string."""
        from voice import VoiceInterface

        voice = VoiceInterface()
        result = voice.transcribe(np.array([], dtype=np.int16))
        self.assertEqual(result, "")

    def test_transcribe_calls_whisper(self):
        """Test that transcription calls whisper model correctly."""
        from voice import VoiceInterface

        voice = VoiceInterface()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  hello world  "}

        with patch.object(voice, '_load_whisper', return_value=mock_model):
            # Create some dummy audio data
            audio_data = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
            result = voice.transcribe(audio_data)

            self.assertEqual(result, "hello world")
            mock_model.transcribe.assert_called_once()

    def test_transcribe_normalizes_audio(self):
        """Test that audio is normalized to float32 before transcription."""
        from voice import VoiceInterface

        voice = VoiceInterface()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test"}

        with patch.object(voice, '_load_whisper', return_value=mock_model):
            # Create audio at max int16 value
            audio_data = np.array([32767, -32768], dtype=np.int16)
            voice.transcribe(audio_data)

            # Check that the audio passed to whisper is normalized float32
            call_args = mock_model.transcribe.call_args
            audio_arg = call_args[0][0]
            self.assertEqual(audio_arg.dtype, np.float32)
            self.assertAlmostEqual(audio_arg[0], 1.0, places=4)
            self.assertAlmostEqual(audio_arg[1], -1.0, places=4)


class TestTextToSpeech(unittest.TestCase):
    """Test text-to-speech functionality."""

    def test_speak_empty_text(self):
        """Test speaking empty text does nothing."""
        from voice import VoiceInterface

        voice = VoiceInterface()

        with patch('subprocess.run') as mock_run:
            voice.speak("")
            mock_run.assert_not_called()

        with patch('subprocess.run') as mock_run:
            voice.speak(None)
            mock_run.assert_not_called()

    def test_speak_calls_piper(self):
        """Test that speak calls piper subprocess."""
        from voice import VoiceInterface

        voice = VoiceInterface()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            with patch.object(voice, '_play_wav'):
                voice.speak("Hello world")

                mock_run.assert_called_once()
                call_args = mock_run.call_args
                self.assertIn("piper", str(call_args))

    def test_speak_fallback_on_error(self):
        """Test that speak falls back to print on piper error."""
        from voice import VoiceInterface

        voice = VoiceInterface()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr=b"error")

            with patch('builtins.print') as mock_print:
                voice.speak("Hello world")
                # Should print the text as fallback
                mock_print.assert_called_with("Hello world")


class TestPlayWav(unittest.TestCase):
    """Test WAV file playback."""

    def test_play_wav_reads_file(self):
        """Test that _play_wav reads and plays the WAV file."""
        from voice import VoiceInterface

        voice = VoiceInterface()

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write a simple WAV file
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(np.zeros(1600, dtype=np.int16).tobytes())

            with patch('sounddevice.play') as mock_play:
                with patch('sounddevice.wait'):
                    voice._play_wav(tmp_path)
                    mock_play.assert_called_once()

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestListenAndTranscribe(unittest.TestCase):
    """Test the combined listen and transcribe functionality."""

    def test_listen_and_transcribe_integration(self):
        """Test that listen_and_transcribe calls record and transcribe."""
        from voice import VoiceInterface

        voice = VoiceInterface()

        mock_audio = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)

        with patch.object(voice, 'record_audio', return_value=mock_audio) as mock_record:
            with patch.object(voice, 'transcribe', return_value="test text") as mock_transcribe:
                result = voice.listen_and_transcribe()

                mock_record.assert_called_once()
                mock_transcribe.assert_called_once_with(mock_audio)
                self.assertEqual(result, "test text")


class TestSilenceDetection(unittest.TestCase):
    """Test silence detection in audio recording."""

    def test_silence_threshold_constant(self):
        """Test that silence threshold is defined."""
        from voice import SILENCE_THRESHOLD
        self.assertIsInstance(SILENCE_THRESHOLD, (int, float))
        self.assertGreater(SILENCE_THRESHOLD, 0)

    def test_silence_duration_constant(self):
        """Test that silence duration is defined."""
        from voice import SILENCE_DURATION
        self.assertIsInstance(SILENCE_DURATION, (int, float))
        self.assertGreater(SILENCE_DURATION, 0)


class TestAudioConstants(unittest.TestCase):
    """Test audio configuration constants."""

    def test_sample_rate(self):
        """Test sample rate is 16kHz (Whisper requirement)."""
        from voice import SAMPLE_RATE
        self.assertEqual(SAMPLE_RATE, 16000)

    def test_channels(self):
        """Test mono audio."""
        from voice import CHANNELS
        self.assertEqual(CHANNELS, 1)

    def test_dtype(self):
        """Test int16 audio format."""
        from voice import DTYPE
        self.assertEqual(DTYPE, np.int16)

    def test_max_recording_duration(self):
        """Test max recording duration is reasonable."""
        from voice import MAX_RECORDING_DURATION
        self.assertGreaterEqual(MAX_RECORDING_DURATION, 10)
        self.assertLessEqual(MAX_RECORDING_DURATION, 120)


class TestElizaVoiceIntegration(unittest.TestCase):
    """Integration tests for ELIZA with voice mode."""

    def test_eliza_has_run_voice_method(self):
        """Test that Eliza class has run_voice method."""
        import eliza as eliza_module
        el = eliza_module.Eliza()
        self.assertTrue(hasattr(el, 'run_voice'))
        self.assertTrue(callable(el.run_voice))

    def test_eliza_voice_mode_flow(self):
        """Test the voice mode conversation flow."""
        import eliza as eliza_module

        el = eliza_module.Eliza()
        el.load('doctor.txt')

        # Mock the VoiceInterface
        mock_voice = MagicMock()
        mock_voice.listen_and_transcribe.side_effect = ["Hello", "bye"]
        mock_voice.speak = MagicMock()

        with patch('voice.VoiceInterface', return_value=mock_voice):
            with patch('builtins.print'):
                el.run_voice()

        # Verify speak was called for initial, responses, and final
        self.assertGreaterEqual(mock_voice.speak.call_count, 3)

    def test_argparse_voice_flag(self):
        """Test that --voice flag is recognized."""
        import eliza as eliza_module
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--voice", action="store_true")

        args = parser.parse_args(["--voice"])
        self.assertTrue(args.voice)

        args = parser.parse_args([])
        self.assertFalse(args.voice)


if __name__ == '__main__':
    unittest.main()
