"""
This module manages the command translation process, transforming spoken or typed natural language
commands into valid, structured JSON actions conforming to predefined Pydantic schemas in actions.py.

To ensure absolute reliability, it uses outlines' grammar state machine (constrained decoding)
to mathematically restrict the model's token output probabilities, forcing it to generate
only JSON matching the schema.

Technical Specifications:
- Uses Unsloth's FastLanguageModel for 4-bit optimized loading on CUDA-capable systems.
- Gracefully falls back to Hugging Face's standard Transformers and PEFT on CPU/MPS (macOS).
- Uses outlines for guided JSON generation via Finite State Machine (FSM) compiled schemas.
- Implements a dynamic caching mechanism to build and compile the outlines generator once,
  maximizing subsequent inference speeds.
- Applies a runtime monkeypatch to outlines' tokenizer hashing to prevent pickle-dill crashes
  caused by Python 3.14 stable ABI differences.
"""

# Import the standard 'warnings' module to programmatically configure, filter, and suppress warning notifications during application runtime execution.
import warnings
# Suppress user-facing warnings matching a specific regular expression pattern related to Hugging Face Transformers parameters conflict.
# Specifically, during constrained decoding or guided generation, outlines sets 'max_new_tokens' which conflicts with the default 'max_length' attribute in the model's generation config, triggering a UserWarning.
# We suppress this UserWarning to prevent console noise, maintaining a clean standard output interface.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*max_new_tokens.*"
)

# Import PyTorch library to check hardware environments and map tensor values
import torch

# Dynamically decide if we can use Unsloth's optimized CUDA kernels, otherwise fall back to Transformers/PEFT
try:
    # Check if a CUDA-enabled GPU is accessible on the system
    if torch.cuda.is_available():
        # Import FastLanguageModel from the unsloth library
        from unsloth import FastLanguageModel
        HAS_UNSLOTH = True
    else:
        # Fall back because we are in a non-CUDA environment like macOS MPS/CPU
        HAS_UNSLOTH = False
except ImportError:
    # Fall back if unsloth library is not installed in the environment
    HAS_UNSLOTH = False

# Import standard Hugging Face components if Unsloth is not active
if not HAS_UNSLOTH:
    # AutoTokenizer and AutoModelForCausalLM are used to download and instantiate weights and vocabularies
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # PeftModel wraps the base causal model with parameter-efficient fine-tuning LoRA adapters
    from peft import PeftModel

# Import standard datetime module to supply dynamic date and time to system context
from datetime import datetime

# Global cache variable to store the outlines JSON generator across inference cycles
_generator = None

# Define the training-aligned system prompt template for routing actions
SYSTEM_TEMPLATE = """
You are a precise macOS system routing core. Your only objective is to translate natural language commands into a single, minimized, valid JSON object.

Strict Rules:
1. Output ONLY raw, valid JSON. Do not include markdown code blocks (```json), conversational text, or explanations.
2. Select the specific "app" and "action" that matches the user's explicit intent.
3. If the command is ambiguous, asks for conversational fluff, or requests an action outside your capabilities, you MUST route to the fallback trigger exactly: {{"app":"router","action":"do-nothing","params":{{}}}}

System Context:
- Current Date: {current_date}
- Current Time: {current_time}
"""

def load_translation_model(base_model_name, adapter_model_name, execution_device):
    """
    Loads the causal language model and tokenizer using the most optimized framework.
    Uses Unsloth FastLanguageModel on CUDA, and falls back to standard HF/PEFT on CPU/MPS.
    """
    # Check if we should use the CUDA-optimized Unsloth loading strategy
    if HAS_UNSLOTH:
        # Load the base model and Lora adapter in 4-bit precision with a 2048 token sequence limit
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = adapter_model_name,
            max_seq_length = 2048,
            dtype = None, # Automatic dtype resolution
            load_in_4bit = True, # Enable 4-bit quantization
        )
        # Configure model parameters for optimized inference mode
        FastLanguageModel.for_inference(model)
    else:
        # Fallback path for local macOS execution without CUDA
        # Instantiate the model's vocabulary and string encoders
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # Load the base model in 16-bit brain float format mapped to execution device (MPS or CPU)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=execution_device
        )
        # Layer the fine-tuned LoRA adapters over the base model
        model = PeftModel.from_pretrained(base_model, adapter_model_name)
        # Set the model to evaluation (eval) mode to disable training dropouts
        model.eval()
        
    # Return both the combined model and its tokenizer back to the orchestrator script
    return model, tokenizer

def translate_command(model, tokenizer, command_text, execution_device):
    """
    Formats the spoken command using ChatML prompts, and passes it to the outlines-guided
    grammar decoder to generate mathematically valid JSON matching the action schemas.
    """
    global _generator
    
    # Initialize and compile the outlines JSON generator if it is not cached
    if _generator is None:
        # Import outlines library to access the structured generation engine
        import outlines
        
        # Import the tokenizer wrapper class from outlines to register token mapping
        from outlines.models.transformers import TransformerTokenizer
        
        # Apply a runtime monkeypatch to the __hash__ method to avoid dill pickle crashes on Python 3.14
        TransformerTokenizer.__hash__ = lambda self: hash(id(self.tokenizer))
        
        # Import the standard reflection module inspect to dynamically read schemas
        import inspect
        # Import BaseModel and RootModel to handle Pydantic validation and schema definitions
        from pydantic import BaseModel, RootModel
        # Import Union to handle multiple type branches programmatically
        from typing import Union
        # Import the project actions module containing macOS schema specifications
        from src import actions
        
        # Query all class definitions in actions.py that subclass Pydantic's BaseModel
        action_classes = [
            obj for name, obj in inspect.getmembers(actions, inspect.isclass)
            if issubclass(obj, BaseModel) and obj is not BaseModel
        ]
        
        # Build a Pydantic RootModel representing the union of all registered action schemas
        ActionUnion = RootModel[Union[*action_classes]]
        
        # Import the outlines model wrapper class for standard transformers
        from outlines.models.transformers import Transformers
        # Wrap our model and tokenizer inside outlines' custom model interface
        outlines_model = Transformers(model, tokenizer)
        
        # Compile the FSM index and initialize the JSON schema guided generator
        _generator = outlines.generate.json(outlines_model, ActionUnion)

    # Retrieve the current local date and time as a datetime object instance using the system's timezone.
    now = datetime.now()
    # Format the retrieved datetime instance into a human-readable, locale-specific date string representation.
    current_date = now.strftime("%A, %Y-%m-%d")
    # Format the current time using 24-hour hour representation (%H), zero-padded minute representation (%M),
    # and zero-padded second representation (%S), producing a string format like "HH:MM:SS" for exact chronological ordering.
    current_time = now.strftime("%H:%M:%S")

    # Format the prompt using the ChatML training layout
    prompt = (
        f"<|im_start|>system\n{SYSTEM_TEMPLATE.format(current_date=current_date, current_time=current_time)}<|im_end|>\n"
        f"<|im_start|>user\n{command_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    # Generate the action using the grammar state machine generator
    result = _generator(prompt)
    
    # Extract the raw, minimized JSON string from the Pydantic RootModel response
    json_action = result.model_dump_json()
    
    # Return the generated action JSON string
    return json_action
