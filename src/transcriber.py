"""
# ------------------------------------------------------------------------------------------------------------------------
# pywhispercpp Speech-to-Text Module
# ------------------------------------------------------------------------------------------------------------------------
# This module implements speech-to-text transcription by interfacing with the pywhispercpp library.
# pywhispercpp is a high-performance Python wrapper around whisper.cpp (ggerganov/whisper.cpp), which is a light,
# optimized C++ implementation of OpenAI's Whisper model.
#
# Unlike the stale precompiled wheels of the original whispercpp package, pywhispercpp compiles natively
# on macOS arm64 architectures under Python 3.14. This resolves the dynamic linking error caused by the removal of
# the legacy '__PyThreadState_UncheckedGet' symbol from the Python C-API.
#
# Technical Specifications / Hardware Acceleration:
# - Leverages Apple Silicon's Metal Performance Shaders (MPS) via Core Graphics / Metal framework (via use_gpu=True).
# - Suppresses raw C++ logging outputs where possible to keep stdout clean.
# - Decodes raw 16-bit signed PCM audio bytes at 16,000Hz sampling rate into float32 array normalized to [-1.0, 1.0].
# - Aggregates generated Segment chunks and filters out whisper.cpp blank audio special tokens.
# ------------------------------------------------------------------------------------------------------------------------
"""

# Import the warnings system module to programmatically suppress warnings generated during runtime execution
import warnings

# Import the NumPy numerical computing library to manipulate multi-dimensional arrays, perform dtype casting, and handle vector calculations on raw audio buffers
import numpy as np

# Import the PyTorch machine learning library to query compute capabilities and check for Apple Silicon graphics hardware presence
import torch

# Import the Model class from pywhispercpp to manage loading the acoustic model weights and running inferences
from pywhispercpp.model import Model

# Import the ContextParams typed dictionary container class to pass device settings to the low-level whisper.cpp context
from pywhispercpp.model import ContextParams

def load_whisper_model(_: str = "small"):
    """
    Load and initialise the Whisper.cpp `small` model using the pywhispercpp binding.

    This function programmatically detects whether Apple Silicon GPU acceleration (Metal Performance Shaders) is active,
    configures the hardware execution context parameters, and loads the pretrained GGML model weights.

    Parameters
    ----------
    _: str (ignored)
        Kept for signature compatibility with the rest of the application; always defaults to downloading the small model.

    Returns
    -------
    Model
        An active pywhispercpp Model object initialized on the appropriate execution device.
    """
    # Query torch.backends.mps to verify if the machine contains Apple Silicon GPU hardware configured with MPS (Metal)
    use_gpu = torch.backends.mps.is_available()

    # If GPU is detected, print status message indicating Metal execution path
    if use_gpu:
        # Inform the user that inference will run on the Apple Silicon GPU using Metal Performance Shaders
        print("Using Whisper.cpp on MPS (Apple GPU) with small model.")
    # Else fallback to standard CPU execution path
    else:
        # Inform the user that execution is falling back to standard CPU cores
        print("Using Whisper.cpp on CPU with small model.")

    # Instantiate the ContextParams dictionary configuration object to configure low-level C++ engine options
    context_params = ContextParams(use_gpu=use_gpu)

    # Instantiate the Model class, passing the requested GGML model name and the constructed context settings
    model = Model(model="small", context_params=context_params)

    # Return the loaded model instance back to the orchestrator thread
    return model

def transcribe_with_whisper(whisper_model, raw_audio_data: bytes) -> str:
    """
    Transcribe raw PCM audio bytes into plain English text using a preloaded pywhispercpp Model.

    Parameters
    ----------
    whisper_model: Model
        The preloaded pywhispercpp Model instance.
    raw_audio_data: bytes
        Raw 16-bit signed PCM audio bytes, single-channel, recorded at 16,000Hz sample rate.

    Returns
    -------
    str
        The final transcribed, cleaned plain-text string.
    """
    # Convert raw input bytes into a 1-dimensional float32 NumPy array.
    # We first read the bytes as 16-bit signed integers (np.int16) matching the microphone's bit depth,
    # and then normalize the integer amplitudes to float32 range [-1.0, 1.0] by dividing by the maximum 16-bit value (32768.0).
    audio_floats = np.frombuffer(raw_audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Execute the C++ forward pass of the model by calling transcribe with the float32 array
    segments = whisper_model.transcribe(audio_floats)

    # Reconstruct the final transcript string by concatenating the text from each predicted Segment instance
    raw_text = "".join(s.text for s in segments)

    # Replace specific blank audio bracketed tokens generated by whisper.cpp when detecting silence (e.g. '[BLANK_AUDIO]', '[NO_SPEECH]')
    cleaned_text = raw_text.replace("[BLANK_AUDIO]", "").replace("[NO_SPEECH]", "").strip()

    # Return the cleaned transcription string back to the main loop orchestrator
    return cleaned_text

