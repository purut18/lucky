"""
This module manages the command translation process. It takes natural language text and uses a large language model (LLM) to convert it into a valid macOS AppleScript (osascript).
Specifically, it wraps Hugging Face's SmolLM2-1.7B-Instruct model with a LoRA (Low-Rank Adaptation) adapter fine-tuned specifically to write AppleScript.

Technical Details:
- Uses the Hugging Face 'transformers' and 'peft' libraries to load base and adapter weights.
- AutoModelForCausalLM is loaded using bfloat16 (16-bit brain floating point format) for memory savings and speed.
- Employs device mapping (using MPS/Metal Performance Shaders on macOS, or fallback CPU).
- Wraps the base model with PeftModel to attach the task-specific LoRA adapter weights.
- Employs greedy generation (do_sample=False) to ensure deterministic translation output.
- Decodes output tokens and crops prompts using token lengths.
"""

# Import the PyTorch library to manage tensors and hardware resources like graphic processors
import torch

# Import AutoTokenizer and AutoModelForCausalLM to download and load our pre-trained AI language model and token decoder
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import PeftModel from the peft library to attach fine-tuning adapter layers (LoRA) to our base language model
from peft import PeftModel

def load_translation_model(base_model_name, adapter_model_name, execution_device):
    """
    Downloads and loads the base tokenizer, base causal LM, and LoRA adapter weights.
    Returns the loaded model and its tokenizer in eval mode.
    """
    # Load the base model's tokenizer to translate words/characters into numbers that the AI understands
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load the base causal language model using 16-bit floating points (bfloat16) on the specified device (MPS or CPU)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=execution_device
    )
    
    # Wrap the base model with our custom LoRA adapter model, which holds the rules for writing macOS command code
    model = PeftModel.from_pretrained(base_model, adapter_model_name)
    
    # Set the model to evaluation (eval) mode to deactivate training behaviors like dropout
    model.eval()
    
    # Return both the combined model and its tokenizer back to the orchestrator script
    return model, tokenizer

def translate_command(model, tokenizer, command_text, execution_device):
    """
    Formats the user's spoken command into a structured chat template,
    runs inference through the LLM, and decodes the result into AppleScript code.
    """
    # Define system instructions and user request as a list of role-play dictionaries
    messages = [
        # Inform the system of its role: acting as a highly precise system command converter
        {"role": "system", "content": "You are an expert system utility that translates natural language commands accurately into macOS osascript execution lines."},
        # Give the model the actual spoken text we want to turn into executable macOS code
        {"role": "user", "content": f"Convert this command to osascript: {command_text}"}
    ]
    
    # Format the message lists into a single text prompt using the model's preferred chat style template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Convert the prompt text into integer tensor coordinates and place them on our device (MPS/CPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(execution_device)
    
    # Disable gradient computations to save computer memory and speed up the generation process
    with torch.no_grad():
        # Ask the model to generate output tokens up to a limit of 128 new tokens, using greedy decoding
        outputs = model.generate(
            # Pass our tokenized inputs as keyword arguments
            **inputs,
            # Set the maximum number of new tokens to create before stopping
            max_new_tokens=128,
            # Set sampling to False to always pick the most likely word (greedy decoding) for absolute stability
            do_sample=False,
            # Pass the end-of-string token identifier to let the generator know when it can stop early
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice the output tokens to keep only the newly generated ones, throwing away the input prompt tokens
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    
    # Translate the generated output tokens back into readable string code, skipping system markers
    osascript_code = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Return the generated AppleScript code string
    return osascript_code
