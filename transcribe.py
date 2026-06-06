"""
This is the main orchestrator script for the Local Speech-To-Text and System Command Execution Utility.
It provides a CLI menu to choose transcription models, initializes input streams, and runs a loop that transcribes voice commands and executes system scripts.

Technical Details:
- Imports modules from the 'src' package: config (env values), audio (mic/keys), transcriber (Vosk/Whisper), translator (LLM), and executor (osascript execution).
- Implements a command-line interface menu to select models (Vosk US, Vosk India, Vosk Hindi, or Faster-Whisper Small/Large).
- Synchronizes Alt/Option keypress events (via audio.recording_started_event/stopped_event) to implement push-to-talk.
- Feeds audio bytes to the selected transcription function, retrieves text, translates it via SmolLM2-LoRA model on MPS or CPU, and executes the output shell-script using subprocess.
"""

# Import the sys module to communicate with standard streams and force output text to show up instantly on screen
import sys

# Import our custom config settings containing paths, devices, and global configuration flags
from src import config

# Import our audio tools to manage microphone streams, push-to-talk states, and keyboard listeners
from src import audio

# Import our transcriber methods to run offline speech recognition using Vosk or Faster-Whisper
from src import transcriber

# Import our translator module to access Hugging Face model loading and AppleScript code generation
from src import translator

# Import our executor function to run the generated AppleScript on our macOS machine via shell subprocesses
from src import executor

def select_mode():
    """
    Displays the CLI menu interface options on the console screen,
    and returns the user's choice selection.
    """
    # Print a line of equals symbols to act as a top border for our CLI header
    print("=" * 60)
    # Print the main header title of our application
    print("      LOCAL SPEECH-TO-TEXT TRANSCRIPTION UTILITY")
    # Print another line of equals symbols to act as a divider line
    print("=" * 60)
    # Print introductory text telling the user to choose an options mode
    print("Please select transcription mode:")
    # Print option number 1: English US using the Small Vosk engine
    print("  [1] English (US) - Small (en-us)")
    # Print option number 2: English US using the Big Vosk engine
    print("  [2] English (US) - Big (vosk-model-en-us-0.22-lgraph)")
    # Print option number 3: English India using the Small Vosk engine
    print("  [3] English (India) - Small (vosk-model-small-en-in-0.4)")
    # Print option number 4: Hindi language using the Vosk engine
    print("  [4] Hindi (hi)")
    # Print option number 5: Parallel English and Hindi small model translation
    print("  [5] Parallel Mode - Small US (English US Small and Hindi)")
    # Print option number 6: Parallel English and Hindi big model translation
    print("  [6] Parallel Mode - Big US (English US Big and Hindi)")
    # Print option number 7: Parallel Indian English and Hindi small model translation
    print("  [7] Parallel Mode - Indian English (English India Small and Hindi)")
    # Print option number 8: Faster-Whisper Small multi-lingual model
    print("  [8] Faster-Whisper - Small Model (Multi-lingual / Auto-detect)")
    # Print option number 9: Faster-Whisper Large high-fidelity model
    print("  [9] Faster-Whisper - Large Model (Multi-lingual / Auto-detect)")
    # Print option number q: Quit the program
    print("  [q] Quit")
    # Print a line of dashes to act as a bottom border for the menu
    print("-" * 60)
    
    # Enter an infinite loop to keep prompting the user until they give a valid input
    while True:
        # Prompt the user for input, clean extra spaces, and convert to lowercase
        choice = input("Enter choice (1-9 or q): ").strip().lower()
        # Verify if the cleaned input is one of the valid menu choices
        if choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'q']:
            # Return the valid choice to the caller function
            return choice
        # If the input was invalid, print an error message and let the loop repeat the prompt
        print("Invalid choice. Please try again.")

def main():
    """
    Main orchestrator function that sets up configuration, downloads models,
    listens for keys, records speech, transcribes, translates, and executes.
    """
    # Call the select_mode function to display the menu and retrieve the user's choice
    choice = select_mode()
    # Check if the user selected 'q' to quit the application
    if choice == 'q':
        # Print exit status message on screen
        print("Exiting...")
        # Stop execution of the function early to shut down the script
        return

    # Initialize empty dictionaries to store our loaded transcription models
    models = {}
    # Initialize empty dictionaries to store our active KaldiRecognizer speech decoders
    recognizers = {}

    # Wrap model loading in a try block to handle network, memory, or download errors gracefully
    try:
        # Check if the user selected options that require the US English Small model (option 1 or parallel option 5)
        if choice in ['1', '5']:
            # Print a progress status message showing we are initializing the English small model
            print("\nLoading English US Small model (auto-downloading if not cached)...")
            # Load the model and recognizer and store them in our dictionaries using 'en_small' as the key
            models['en_small'], recognizers['en_small'] = transcriber.load_vosk_model(
                "en-us",
                config.SAMPLE_RATE,
                is_lang=True
            )
            # Print success status message showing model setup was successful
            print("✓ English US Small model loaded.")

        # Check if the user selected options that require the US English Big model (option 2 or parallel option 6)
        if choice in ['2', '6']:
            # Print a progress status message showing we are initializing the English big model
            print("\nLoading English US Big model (vosk-model-en-us-0.22-lgraph) (auto-downloading if not cached)...")
            # Load the model and recognizer and store them in our dictionaries using 'en_big' as the key
            models['en_big'], recognizers['en_big'] = transcriber.load_vosk_model(
                "vosk-model-en-us-0.22-lgraph",
                config.SAMPLE_RATE,
                is_lang=False
            )
            # Print success status message showing model setup was successful
            print("✓ English US Big model loaded.")

        # Check if the user selected options that require the Indian English Small model (option 3 or parallel option 7)
        if choice in ['3', '7']:
            # Print a progress status message showing we are initializing the Indian English small model
            print("\nLoading English India Small model (vosk-model-small-en-in-0.4) (auto-downloading if not cached)...")
            # Load the model and recognizer and store them in our dictionaries using 'en_in' as the key
            models['en_in'], recognizers['en_in'] = transcriber.load_vosk_model(
                "vosk-model-small-en-in-0.4",
                config.SAMPLE_RATE,
                is_lang=False
            )
            # Print success status message showing model setup was successful
            print("✓ English India Small model loaded.")

        # Check if the user selected options that require the Hindi model (option 4, or parallel options 5, 6, 7)
        if choice in ['4', '5', '6', '7']:
            # Print a progress status message showing we are initializing the Hindi model
            print("\nLoading Hindi model (auto-downloading if not cached)...")
            # Load the model and recognizer and store them in our dictionaries using 'hi' as the key
            models['hi'], recognizers['hi'] = transcriber.load_vosk_model(
                "hi",
                config.SAMPLE_RATE,
                is_lang=True
            )
            # Print success status message showing model setup was successful
            print("✓ Hindi model loaded.")

        # Check if the user selected Faster-Whisper options (option 8 for Small, 9 for Large)
        if choice in ['8', '9']:
            # Determine Whisper model scale to download: 'small' for choice 8, 'large-v3' for choice 9
            model_size = "small" if choice == '8' else "large-v3"
            # Print progress status message showing Whisper model loader is starting
            print(f"\nLoading Faster-Whisper {model_size} model (auto-downloading if not cached)...")
            # Load the model and store it in our models dictionary using 'whisper' as the key
            models['whisper'] = transcriber.load_whisper_model(model_size)
            # Print success status message showing model setup was successful
            print(f"✓ Faster-Whisper {model_size} model loaded.")

        # Print progress status message showing command translator LLM loading is starting
        print("\nLoading command translation model (SmolLM2-1.7B-Instruct + LoRA)...")
        # Print message showing which compute hardware device (MPS/CPU) PyTorch is using
        print(f"Using device: {config.HF_DEVICE} for command translation")
        # Call translator module function to build base SmolLM2 model with PEFT adapters loaded
        hf_model, hf_tokenizer = translator.load_translation_model(
            config.BASE_MODEL,
            config.ADAPTER_MODEL,
            config.HF_DEVICE
        )
        # Print success status message showing LLM setup was successful
        print("✓ Command translation model loaded successfully.")

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
        # Setup human-friendly label for CLI logs when using Faster-Whisper engine options
        model_label = "Faster-Whisper Small" if choice == '8' else "Faster-Whisper Large"
        
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
                sys.stdout.flush()
                # Skip remainder of the pipeline and restart loop waiting for next press event
                continue
                
            # Initialize empty text strings to store single and parallel language transcription transcripts
            text = ""
            text_en = ""
            
            # Match user select choice to run transcription through the appropriate speech recognizer
            if choice == '1':
                # Transcribe speech using US English Small offline model
                text = transcriber.transcribe_with_vosk(recognizers['en_small'], raw_audio)
                # Print result string to stdout
                sys.stdout.write(f"\r\033[K[English US Small] Final: {text}\n")
                sys.stdout.flush()
                
            elif choice == '2':
                # Transcribe speech using US English Big offline model
                text = transcriber.transcribe_with_vosk(recognizers['en_big'], raw_audio)
                # Print result string to stdout
                sys.stdout.write(f"\r\033[K[English US Big] Final: {text}\n")
                sys.stdout.flush()
                
            elif choice == '3':
                # Transcribe speech using Indian English Small offline model
                text = transcriber.transcribe_with_vosk(recognizers['en_in'], raw_audio)
                # Print result string to stdout
                sys.stdout.write(f"\r\033[K[English India Small] Final: {text}\n")
                sys.stdout.flush()
                
            elif choice == '4':
                # Transcribe speech using Hindi offline model
                text = transcriber.transcribe_with_vosk(recognizers['hi'], raw_audio)
                # Print result string to stdout
                sys.stdout.write(f"\r\033[K[Hindi] Final: {text}\n")
                sys.stdout.flush()
                
            elif choice in ['5', '6', '7']:
                # For parallel options, map the English recognizer key based on menu choice
                if choice == '5':
                    # Select the English US Small recognizer
                    rec_en = recognizers['en_small']
                    # Label output text as US Small
                    label_en = "[English US Small]"
                elif choice == '6':
                    # Select the English US Big recognizer
                    rec_en = recognizers['en_big']
                    # Label output text as US Big
                    label_en = "[English US Big]"
                else:
                    # Select the English Indian Small recognizer
                    rec_en = recognizers['en_in']
                    # Label output text as Indian English Small
                    label_en = "[English IN Small]"
                    
                # Store reference to Hindi recognizer model
                rec_hi = recognizers['hi']
                
                # Run English transcription on our recorded audio bytes
                text_en = transcriber.transcribe_with_vosk(rec_en, raw_audio)
                # Run Hindi transcription on our recorded audio bytes
                text_hi = transcriber.transcribe_with_vosk(rec_hi, raw_audio)
                
                # Output English transcription final text result to stdout
                sys.stdout.write(f"\r\033[K{label_en} Final: {text_en}\n")
                # Output Hindi transcription final text result to stdout
                sys.stdout.write(f"[Hindi] Final: {text_hi}\n")
                sys.stdout.flush()
                
            elif choice in ['8', '9']:
                # Run transcription through Faster-Whisper small or large models using recorded bytes
                text = transcriber.transcribe_with_whisper(models['whisper'], raw_audio)
                # Print transcription output to stdout
                sys.stdout.write(f"\r\033[K[{model_label}] Final: {text}\n")
                sys.stdout.flush()
                
            # Filter and store the raw text transcript we want to translate into terminal command
            command_text = ""
            # If parallel mode is active, use the English transcription output
            if choice in ['5', '6', '7']:
                # Set command text input to English transcript
                command_text = text_en.strip()
            # If standard mode is active, use primary transcription output
            else:
                # Set command text input to primary language transcript
                command_text = text.strip()
 
            # Check if our transcription result is non-empty before processing
            if command_text:
                # Update status showing the translation process has started
                sys.stdout.write("\r\033[K[Translating command...]")
                sys.stdout.flush()
                
                # Call translator module to format prompts and generate AppleScript code via LLM
                osascript_code = translator.translate_command(
                    hf_model,
                    hf_tokenizer,
                    command_text,
                    config.HF_DEVICE
                )
                
                # Output user spoken command on screen
                sys.stdout.write(f"\r\033[K[Command]: {command_text}\n")
                # Output generated AppleScript command code on screen
                sys.stdout.write(f"[Generated Code]: {osascript_code}\n")
                sys.stdout.flush()
                
                # Check if generated script is not empty before attempting execution
                if osascript_code:
                    # Execute generated AppleScript code on terminal via executor subprocess module
                    executor.execute_osascript(osascript_code)
 
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
