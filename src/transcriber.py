"""
This module manages the speech-to-text engines: Vosk (a Kaldi-based offline speech recognizer) and Faster-Whisper (a highly optimized version of OpenAI's Whisper model running on CTranslate2).
It is capable of loading multiple model paths/types (e.g., small, big, localized language models like Indian English and Hindi) and performing inference on raw input buffers.

Technical Details:
- Uses the 'vosk' package to run local voice recognition. SetLogLevel(-1) is used to silence C-level output.
- Employs 'faster_whisper.WhisperModel' for high-fidelity speech recognition, running on the CPU with 8-bit integer quantization (int8) to reduce memory usage.
- Standardizes audio data preprocessing: Faster-Whisper requires normalized floating-point PCM samples (-1.0 to 1.0), which are converted from 16-bit signed integers (int16) by dividing by 32768.0.
- Implements VAD (Voice Activity Detection) parameters within Whisper to ignore silent segments and avoid processing noise.
"""

# Import the JSON package to parse Vosk output strings (which return as text strings in JSON format)
import json

# Import the warnings module to disable non-critical alerts like math overflow warnings
import warnings

# Import the numpy library to transform audio buffers from integer arrays into decimal-point float arrays
import numpy as np

# Import Model, KaldiRecognizer, and SetLogLevel from vosk to initialize our local offline language engines
from vosk import Model, KaldiRecognizer, SetLogLevel

# Suppress Vosk's internal low-level debugging messages to keep our CLI interface neat and clean
SetLogLevel(-1)

def load_vosk_model(model_identifier, sample_rate, is_lang=True):
    """
    Loads a specific Vosk model and returns the Model object and its KaldiRecognizer.
    """
    # If the identifier is a language shorthand (like 'en-us' or 'hi'), load it using the lang parameter
    if is_lang:
        # Load the model from local cache or auto-download from the internet if not already stored
        model = Model(lang=model_identifier)
    # If the identifier is a custom directory name (like 'vosk-model-en-us-0.22-lgraph'), load by name
    else:
        # Load the model by locating the exact model folder name in the cache/installation directories
        model = Model(model_name=model_identifier)
    # Instantiate a KaldiRecognizer object, which takes the loaded model and matches it against our recording speed
    recognizer = KaldiRecognizer(model, sample_rate)
    # Return both the model and the recognizer as a tuple to the calling function
    return model, recognizer

def load_whisper_model(model_size):
    """
    Loads the Faster-Whisper model for a given model size (e.g., 'small' or 'large-v3').
    """
    # Suppress the numerical instability RuntimeWarning from faster_whisper during silent frames to avoid terminal clutter
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="faster_whisper")
    # Import the WhisperModel class from the faster-whisper package dynamically when this engine is requested
    from faster_whisper import WhisperModel
    # Load the WhisperModel on CPU, running on 8-bit integers (int8) to maximize performance and save computer RAM
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # Return the initialized model object so it can perform translations
    return model

def transcribe_with_vosk(recognizer, raw_audio_data):
    """
    Transcribes raw PCM bytes audio using a loaded Vosk KaldiRecognizer.
    """
    # Reset the internal states of the recognizer so old words do not bleed into our new voice command
    recognizer.Reset()
    # Feed the raw PCM audio bytes into the recognizer engine to analyze the sound waveforms
    recognizer.AcceptWaveform(raw_audio_data)
    # Fetch the final text prediction as a JSON string, then parse it into a standard Python dictionary
    result_dict = json.loads(recognizer.FinalResult())
    # Extract and return the transcribed text value from the dictionary, defaulting to an empty string if not found
    return result_dict.get("text", "")

def transcribe_with_whisper(whisper_model, raw_audio_data):
    """
    Converts raw PCM audio bytes into float32 format, then transcribes using Faster-Whisper.
    """
    # Convert the raw bytes buffer into a 16-bit integer numpy array, then scale it to float32 decimal range (-1.0 to 1.0)
    audio_floats = np.frombuffer(raw_audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    # Run the transcription engine on our float array, setting beam size to 5 and filtering out silence blocks
    segments, info = whisper_model.transcribe(
        audio_floats,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        language="en"
    )
    # Combine the text attributes of all transcribed speech segments into a single space-separated string
    transcribed_text = " ".join([segment.text for segment in segments]).strip()
    # Return the clean transcribed text string
    return transcribed_text
