"""
This config module is responsible for setting up and managing application-wide settings and environment variables.
It uses the python-dotenv library to read values from a local '.env' file, which prevents us from hardcoding
sensitive API tokens (such as the Hugging Face HF_TOKEN or OpenAI API Keys) directly into the source code repository.

Technical Details:
- Uses python-dotenv's load_dotenv() to inject key-value pairs from '.env' into os.environ.
- Determines the appropriate PyTorch computing device (MPS for Apple Silicon hardware acceleration, or CPU).
- Exports variables for transcription parameters, model names, and system paths.
- Developers can modify settings here to change the base model, sample rates, or default devices.
"""

# Import the built-in system library 'os' so we can talk to the computer and access environment variables
import os

# Import 'load_dotenv' from the 'dotenv' library, which reads a text file named '.env' and makes its variables active
from dotenv import load_dotenv

# Import the 'torch' library, which is a tool used for running AI models and checking computer graphics processors
import torch

# Call the load_dotenv function to read the '.env' file at the start of our program, putting its settings into os.environ
load_dotenv()

# Retrieve the Hugging Face token from our environment variables and save it to 'HF_TOKEN' for loading gated models
HF_TOKEN = os.environ.get("HF_TOKEN")

# Retrieve the OpenAI API Key from our environment variables and save it to 'OPENAI_API_KEY' if we need it later
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Define the base command translation model name from Hugging Face that we want to download and run
BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Define the adapter model name, which holds extra training info to teach our model how to write macOS command scripts
ADAPTER_MODEL = "thakkar-puru/smollm2-osascript-lora"

# Set the sound recording rate to 16000 Hertz, which is the standard speed that speech recognition models expect
SAMPLE_RATE = 16000

# Check if this Mac has an Apple Silicon chip (MPS) that can run the translation model faster, otherwise use the regular CPU
HF_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
