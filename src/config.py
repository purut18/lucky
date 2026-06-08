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

# Configure the specific Large Language Model identifier string for the background agent to consume.
# This target model name directs the OpenAI SDK to route the API request to the reasoning-optimized model.
LLM_AGENT_MODEL = "gpt-5.4"

# Specify the target level of reasoning effort (or depth of chain-of-thought processing) for the agent.
# This variable is passed directly to the Chat Completions API to control the compute allocation for thinking tokens.
LLM_AGENT_REASONING_EFFORT = "medium"

# Define the baseline system prompt used to align and initialize the context of the reasoning chat completion.
# This instruction set explicitly directs the model to answer the query in markdown format and specify a target filename.
LLM_AGENT_SYSTEM_PROMPT = """
You are a highly capable AI research assistant.
    
# Task
    - Your objective is to provide a comprehensive, detailed answer to the user in markdown format.
        Use all markdown features to create a properly structured response with headings, sources, tables, etc. Whatever is needed to answer the user.
    - Along with the markdown content, you must determine a short, descriptive title for the response.
        - Without spaces or file extension, e.g., 'MarketAnalysis' or 'CodeReview'.
    - Always use the internet to ground your results and provide sources for any claims you make.
    - Try to keep the response concise but detailed and accurate.
"""

LLM_AGENT_USE_INTERNET = True


# Define the base command translation model name from Hugging Face that we want to download and run
BASE_MODEL = "unsloth/Qwen2.5-Coder-1.5B-Instruct"

# Define the adapter model name, which holds extra training info to teach our model how to write macOS command scripts
ADAPTER_MODEL = "models/cmd_to_action_model_3"

# Set the sound recording rate to 16000 Hertz, which is the standard speed that speech recognition models expect
SAMPLE_RATE = 16000

# Check if this Mac has an Apple Silicon chip (MPS) that can run the translation model faster, otherwise use the regular CPU
HF_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
