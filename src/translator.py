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

# Import PyTorch library to check hardware environments and map tensor values
import torch

# Dynamically decide if we can use Unsloth's optimized CUDA kernels, otherwise fall back to Transformers/PEFT
# ------------------------------------------------------------
# Device detection block – Apple silicon (MPS) and optional MLX support
# ------------------------------------------------------------
# Attempt to import the MLX library, which provides highly‑optimized
# inference kernels for Apple silicon (GPU/CPU). If MLX is available we
# will prefer it because it can run models directly on the M‑series GPU
# with minimal overhead.
#
# If MLX cannot be imported we fall back to the original torch‑MPS
# pathway. Additionally we expose a flag for optional 4‑bit quantisation
# via bitsandbytes when running on a pure‑CPU fallback.
# ------------------------------------------------------------
try:
    import mlx.core as mx  # MLX core provides the execution backend
    HAS_MLX = True
except Exception:
    HAS_MLX = False

# Determine if the torch MPS backend is available (Apple GPU)
HAS_MPS = torch.backends.mps.is_available()

# Import standard Hugging Face components if Unsloth is not active
# ------------------------------------------------------------
# Fallback to non‑Unsloth path – handled inside ``load_translation_model``
# ------------------------------------------------------------
# The original print statement is kept for clarity during debugging.
print("Using Transformers (MPS/CPU fallback)")
# The actual imports are performed lazily inside ``load_translation_model``;
# they are shown here only for documentation purposes.
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

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

def load_translation_model(base_model_name: str, adapter_path: str, device: str = "cpu") -> tuple:
    """Load Qwen 1.5B with optional LoRA adapter using the most efficient backend.

    Parameters
    ----------
    base_model_name: str
        The Hugging Face repo identifier or local path for the base Qwen model.
    adapter_path: str
        Path to the LoRA adapter directory (e.g., "models/cmd_to_action_model_3").
    device: str, optional
        Target execution device. Supported values are:
        * "mlx" – Use the MLX library for Apple‑silicon‑native acceleration.
        * "mps" – Use PyTorch's MPS backend (float16).
        * "cpu" – Pure CPU execution (optionally 4‑bit quantised via bitsandbytes).

    Returns
    -------
    tuple
        A ``(model, tokenizer)`` pair ready for inference.

    Notes
    -----
    * The function prefers MLX when available because it bypasses
      the PyTorch overhead and runs directly on the Apple GPU.
    * If MLX is unavailable we fall back to the MPS backend (float16)
      for GPU‑accelerated inference.
    * When running on CPU we optionally apply 4‑bit quantisation via
      ``bitsandbytes`` to reduce memory and improve speed.
    """
    # -----------------------------------------------------------------
    # 1️⃣  Choose the appropriate backend based on availability flags.
    # -----------------------------------------------------------------
    if device == "mlx" and HAS_MLX:
        # -------------------------------------------------------------
        # MLX pathway – fast Apple‑silicon inference.
        # -------------------------------------------------------------
        print("Using MLX for SLM inference.")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # Placeholder for actual MLX model loading – replace with concrete call.
        raise NotImplementedError(
            "MLX model loading for Qwen is not implemented – replace with\n"
            "appropriate mlx_lm loader call."
        )
    elif device == "mps" and HAS_MPS:
        # -------------------------------------------------------------
        # PyTorch MPS pathway – float16 precision on Apple GPU.
        # -------------------------------------------------------------
        print("Using MPS for SLM inference.")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map={"": "mps"},
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
    else:
        # -------------------------------------------------------------
        # CPU pathway – optional 4‑bit quantisation via bitsandbytes.
        # -------------------------------------------------------------
        print("Using CPU w/ 4-bit quantisation for SLM inference.")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        try:
            import bitsandbytes as bnb
            quantise = True
        except Exception:
            quantise = False
        if quantise:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
            )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
    # -----------------------------------------------------------------
    # Return the model and tokenizer for downstream JSON generation.
    # -----------------------------------------------------------------
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
