"""
This is the main orchestrator script for the Local Speech-To-Text and System Command Execution Utility.
It initializes input streams, boots up the default speech-to-text model (Faster-Whisper Small) and command translation model,
and enters a loop that captures user speech (via Alt/Option hold-to-talk key binds), transcribes it, translates it to JSON-structured system tasks,
and runs the resulting generated macOS scripts.

Technical Details / Specifications:
- Imports: Core system audio utilities, Faster-Whisper transcriber module, Qwen-based PEFT LoRA command translation adapter, macOS bash/osascript executor.
- Keyboard monitoring: Utilizes background keyboard polling listeners via Python's system threading primitives to detect the Option/Alt hotkey states.
- Local Models: Loads a pre-trained Faster-Whisper small-size acoustic model configured for INT8 quantization.
- Translation Architecture: Uses a 1.5B parameters instruction-tuned Coder base model customized via LoRA checkpoints.
"""

# Import the sys module to communicate with standard streams and force output text to show up instantly on screen
import sys

# Import our custom config settings containing paths, devices, and global configuration flags
from src import config

# Import our audio tools to manage microphone streams, push-to-talk states, and keyboard listeners
from src import audio

# Import our transcriber methods to run offline speech recognition using Faster-Whisper
from src import transcriber

# Import our translator module to access Hugging Face model loading and AppleScript code generation
from src import translator

# Import our executor function to run the generated AppleScript on our macOS machine via shell subprocesses
from src import executor

# Import our bridge module to convert Pydantic action models to macOS terminal/AppleScript commands
from src import bridge

def main():
    """
    Main orchestrator function that sets up configuration, downloads models,
    listens for keys, records speech, transcribes, translates, and executes.
    
    This function acts as the main program driver, initializing all necessary subsystems
    and maintaining the event-driven transcription-to-execution pipeline loop.
    """
    # Initialize an empty dictionary to store the Whisper model instances in-memory
    models = {}

    # Wrap model loading in a try block to handle network, memory, or download errors gracefully
    try:
        # Define model scale as 'small' for our default Faster-Whisper transcription engine
        model_size = "small"
        
        # Print a progress status message showing we are initializing the Whisper model
        print(f"\nLoading Faster-Whisper {model_size} model (auto-downloading if not cached)...")
        
        # Load the model and store it in our models dictionary using 'whisper' as the key
        models['whisper'] = transcriber.load_whisper_model(model_size)
        
        # Print success status message showing model setup was successful
        print(f"✓ Faster-Whisper {model_size} model loaded.")

        # Print progress status message showing command translator LLM loading is starting
        print(f"\nLoading command translation model ({config.BASE_MODEL} + LoRA)...")
        
        # Print message showing which compute hardware device (MPS/CPU) PyTorch is using
        print(f"Using device: {config.HF_DEVICE} for command translation")
        
        # Call translator module function to build base model with PEFT adapters loaded
        hf_model, hf_tokenizer = translator.load_translation_model(
            config.BASE_MODEL,
            config.ADAPTER_MODEL,
            config.HF_DEVICE
        )
        # Print success status message showing LLM setup was successful
        print("✓ Command translation model loaded successfully.")

        # Print progress status message showing we are performing a dummy inference pass to compile schemas and warm up PyTorch device caches
        print("Warming up command translation model (compiling schemas & tracing device kernels)...")
        # Invoke a dummy translate_command execution pass to construct outlines FSM and trace PyTorch MPS/CPU kernels in advance
        translator.translate_command(hf_model, hf_tokenizer, "warmup", config.HF_DEVICE)
        # Print success status message showing model warmup has successfully completed
        print("✓ Command translation model warmed up and ready.")

    # Handle model loading failure exceptions (such as network timeout, out of disk space)
    except Exception as e:
        # Print error details on terminal
        print(f"\nFailed to load model(s): {e}")
        # Print troubleshooting steps for internet connectivity and firewall blocks
        print("Please check your internet connection and verify that Python is allowed network access.")
        # End the function early because system cannot work without loaded speech/LLM engines
        return

    # Print structural header boundaries to separate loading messages from active audio capture interface
    print("\n" + "=" * 60)
    # Explain how to control the audio recorder: holding down the alt/option key
    print("              PUSH-TO-TALK TRANSCRIPTION MODE")
    # Explain what action happens on press (recording) and release (transcription)
    print("       Hold [Option / Alt] key to record, release to transcribe.")
    # Explain how to close the script safely using Ctrl+C
    print("       Press Ctrl+C to exit.")
    # Print closing menu boundary line
    print("=" * 60 + "\n")

    # Start the background keyboard listener by calling the configuration setup function
    try:
        # Store the listener instance so that we can shut it down gracefully when the program exits
        listener = audio.setup_keyboard_listener()
    # Handle exceptions if system Accessibility and Input Monitoring permissions are missing
    except Exception as e:
        # Print warning detailing keyboard listener initialization failure
        print(f"\nWarning: Could not start keyboard listener: {e}")
        # Instruct user that macOS security blocks background input capture by default
        print("On macOS, this script requires Accessibility and Input Monitoring permissions")
        # Detail exactly where in macOS settings the user needs to activate permissions
        print("for your Terminal or IDE app (System Settings > Privacy & Security).")
        # Tell the user to restart the application after activating permissions
        print("Please grant these permissions and restart the script.")
        # End the function early because keyboard input cannot function without permissions
        return

    # Enter the main recording loop block, capturing exceptions to clean up properly on crash/interrupt
    try:
        # Define the human-friendly label for logs corresponding to our Whisper Small engine choice
        model_label = "Faster-Whisper Small"
        
        # Run the recording loop indefinitely until the user cancels via KeyboardInterrupt
        while True:
            # Block the loop here and wait for the user to press Alt/Option (started event is fired)
            audio.recording_started_event.wait()
            # Reset the started event flag so it blocks on the next loop iteration
            audio.recording_started_event.clear()
            # Reset the stopped event flag to prepare for the end of the recording segment
            audio.recording_stopped_event.clear()
            
            # Start mic recording, capturing audio blocks into memory, returning raw PCM bytes on release
            raw_audio = audio.record_audio(config.SAMPLE_RATE)
            
            # Write status to stdout showing the transcription phase has begun
            sys.stdout.write("\r\033[K[Transcribing...]")
            # Flush stdout to force the status text to be displayed on screen immediately
            sys.stdout.flush()
            
            # Verify that the returned audio buffer is not empty
            if not raw_audio:
                # Log that no speech was captured on the screen
                sys.stdout.write("\r\033[K[No audio recorded]\n")
                # Flush stdout to guarantee the print line is rendered instantly
                sys.stdout.flush()
                # Skip remainder of the pipeline and restart loop waiting for next press event
                continue
                
            # Transcribe the raw audio bytes stream using our preloaded Whisper model weights
            text = transcriber.transcribe_with_whisper(models['whisper'], raw_audio)
            
            # Output transcription final text result to stdout
            sys.stdout.write(f"\r\033[K[{model_label}] Final: {text}\n")
            # Flush the standard out file buffer to display text in real-time
            sys.stdout.flush()
            
            # Clean trailing and leading whitespaces from the transcribed text command before processing
            command_text = text.strip()
 
            # Check if our transcription result is non-empty before processing
            if command_text:
                # Update status showing the translation process has started
                sys.stdout.write("\r\033[K[Translating command...]")
                # Flush stdout to present the translation message in terminal
                sys.stdout.flush()
                
                # Call translator module to format prompts and generate structured JSON action via LLM
                json_action_str = translator.translate_command(
                    hf_model,
                    hf_tokenizer,
                    command_text,
                    config.HF_DEVICE
                )
                
                # Output user spoken command on screen
                sys.stdout.write(f"\r\033[K[Command]: {command_text}\n")
                # Output generated JSON action schema on screen
                sys.stdout.write(f"[Generated JSON]: {json_action_str}\n")
                # Flush output stream to sync print operations
                sys.stdout.flush()
                
                try:
                    # Parse the predicted JSON string into a validated Pydantic model instance
                    action_obj = bridge.action_from_json(json_action_str)
                    
                    # Convert the structured action object into an executable terminal command/AppleScript
                    terminal_cmd = bridge.translate_action_to_command(action_obj)
                    
                    # If translation produced an executable terminal command, run it
                    if terminal_cmd:
                        # Write execution header log of the terminal commands to execute
                        sys.stdout.write(f"[Generated Command/Script]:\n{terminal_cmd}\n")
                        # Sync standard output streams
                        sys.stdout.flush()
                        # Execute the translated command on terminal via executor subprocess module
                        executor.execute_osascript(terminal_cmd)
                    # If action object translated into an empty script, log as skipped
                    else:
                        # Print skip notice
                        sys.stdout.write("[Action skipped / Not implemented for script execution]\n")
                        # Flush stdout
                        sys.stdout.flush()
                # Catch failures in parsing JSON, validation issues or bridge-to-OS command translation
                except Exception as execution_err:
                    # Log any errors during JSON parsing, schema validation, or command translation
                    sys.stdout.write(f"Failed to parse or execute action: {execution_err}\n")
                    # Flush output stream
                    sys.stdout.flush()
 
    # Catch manual user interruption keys (Ctrl+C) to exit program cleanly without stack traces
    except KeyboardInterrupt:
        # Print shutdown messages on screen
        print("\n\nStopping key listener and microphone stream. Goodbye!")
    # Catch any general program exceptions to prevent raw shell crash output
    except Exception as e:
        # Print general error description
        print(f"\nError: {e}")
    # Always perform final hooks release actions before closing down completely
    finally:
        # Stop background keyboard listener thread monitoring keyboard events
        listener.stop()

# Check if this file is run directly as the main execution script
if __name__ == "__main__":
    # Start the program logic by invoking the main function
    main()
