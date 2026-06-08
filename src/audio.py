"""
This module handles audio recording from the system's microphone, captures keyboard events to support push-to-talk functionality,
and supports native audio file playback.
It listens for the Alt/Option key to start and stop audio recording, and records the raw input into standard format bytes.

Technical Details:
- Uses the pynput library to monitor global key presses and releases of the Alt key.
- Utilizes threading.Event signals to synchronize key listener events with the main thread.
- Employs sounddevice.RawInputStream to capture mono, 16-bit PCM integer audio stream data at a given sample rate (e.g., 16000Hz).
- Operates a callback-driven buffer queue to record chunked mic audio into memory safely without audio dropouts.
- Provides a global, cross-component interface to trigger synchronous/asynchronous audio file playback on macOS using system utilities.
"""

# Import the sys library to output messages and flush text directly to the console output
import sys

# Import the threading module to support multi-threaded events and thread-safe signaling
import threading

# Import the sounddevice library to interact with external/internal sound cards and microphones
import sounddevice as sd

# Import the keyboard module from pynput to capture real-time keyboard inputs outside the console window
from pynput import keyboard

# Import the os library to handle file paths, check file existence, and perform file system operations
import os

# Import the subprocess library to execute system commands and spawn child processes
import subprocess

# Define a global boolean flag that indicates whether the system is currently recording sound
is_recording = False

# Create a threading Event that will wake up threads when the user starts holding down the recording key
recording_started_event = threading.Event()

# Create a threading Event that will signal other threads when the user stops holding down the recording key
recording_stopped_event = threading.Event()

def on_press(key):
    """
    Callback function that triggers whenever any keyboard key is pressed down.
    If the Alt/Option key is pressed, we flip our recording flag to True and wake up the recorder.
    """
    # Use the global variables so that changes to these settings are seen by the whole program
    global is_recording
    # Set up error protection to prevent crashes if key comparison fails (e.g., with special keys)
    try:
        # Check if the key that the user pressed down is the 'alt' key (Option key on Mac computers)
        if key == keyboard.Key.alt:
            # If we were not already recording audio, then we should start recording now
            if not is_recording:
                # Set our global recording state to True to signify recording is active
                is_recording = True
                # Set the started event to True, which wakes up any thread waiting to start recording
                recording_started_event.set()
    # If any error occurs inside the key listener (like invalid key types), ignore it and do not crash
    except Exception:
        # Pass silently since we do not want key capture errors to interrupt our user
        pass

def on_release(key):
    """
    Callback function that triggers whenever any keyboard key is released/let go.
    If the Alt/Option key is released, we flip our recording flag to False and signal the recorder to stop.
    """
    # Use the global variables so that changes to these settings are seen by the whole program
    global is_recording
    # Set up error protection to prevent crashes if key comparison fails (e.g., with special keys)
    try:
        # Check if the key that the user released/let go is the 'alt' key (Option key on Mac computers)
        if key == keyboard.Key.alt:
            # If we were currently recording audio, then we should stop recording now
            if is_recording:
                # Set our global recording state to False to signify recording has ended
                is_recording = False
                # Set the stopped event to True, which tells other threads that recording is done
                recording_stopped_event.set()
    # If any error occurs inside the key listener (like invalid key types), ignore it and do not crash
    except Exception:
        # Pass silently since we do not want key capture errors to interrupt our user
        pass

def setup_keyboard_listener():
    """
    Creates and starts a background thread that listens for keys globally.
    Requires Accessibility permissions on macOS.
    """
    # Create a new keyboard listener pointing to our custom key press and key release functions
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    # Start the listener running in the background as a separate thread so it does not block the program
    listener.start()
    # Return the listener object so the parent program can stop it when exiting
    return listener

def record_audio(sample_rate):
    """
    Listens to the system's microphone and records raw audio data as long as is_recording is True.
    """
    # Print a clean status message on the terminal showing we are actively recording sound
    sys.stdout.write("\r\033[K[Recording... Speak now]")
    # Flush the print buffer immediately to display the message right away without delaying
    sys.stdout.flush()
    # Create an empty list to store incoming chunks of audio data as bytes in memory
    audio_data_list = []
    
    # Define a callback function that the audio driver calls each time it has new audio chunks
    def audio_callback(indata, frames, time, status):
        # Check if the sound driver reports any problems (like buffers being overloaded or underloaded)
        if status:
            # Print the driver status warnings to the standard error console for troubleshooting
            print(f"Status callback: {status}", file=sys.stderr)
        # Convert the audio buffer input bytes and append them directly to our audio list
        audio_data_list.append(bytes(indata))
        
    # Open a RawInputStream to record audio with specific sample rate, block size, 1 channel, and int16 format
    with sd.RawInputStream(samplerate=sample_rate, blocksize=4000, dtype='int16',
                           channels=1, callback=audio_callback):
        # Keep looping and keeping the stream open as long as the user holds down the record key
        while is_recording:
            # Sleep the thread for 50 milliseconds to avoid wasting CPU cycles while waiting
            sd.sleep(50)
            
    # Return the combined audio list converted into a single big string of raw audio bytes
    return b"".join(audio_data_list)

def play_audio(file_path: str, wait: bool = False) -> None:
    """
    Universal global utility function to play an audio file on macOS via the native system CLI utility 'afplay'.
    This function can be imported and executed from any component of the application to deliver audio feedback.

    Technical Details:
    - Path Resolution: Leverages 'os.path' functions to calculate absolute file system paths.
      If a relative path is passed, it is combined with the project root directory (inferred from the position of 'audio.py').
    - Validation: Performs existence checks via 'os.path.exists' to handle missing files and prevent system-level exceptions.
    - Subprocess Spawning: Calls macOS's 'afplay' utility through the standard 'subprocess' module.
    - Process Execution Modes:
      - Asynchronous (wait=False): Utilizes 'subprocess.Popen' to fork a child process that runs concurrently without blocking Python's main GIL.
      - Synchronous (wait=True): Utilizes 'subprocess.run' to block the calling thread until the child process terminates and returns.
    - Exception Isolation: Uses broad try-except block trapping to prevent audio failures from crashing the orchestration engine.

    Args:
        file_path (str): The absolute or project-relative file system path to the target audio file.
        wait (bool): Flag determining whether the execution blocks the current thread until playback completes.
    """
    # Wrap the entire execution block in a try-except structure to catch process, file system, or system-level exceptions
    try:
        # Determine the absolute directory path where this source file (audio.py) is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Compute the root directory of the project by navigating one directory level up from the 'src' folder
        project_root = os.path.dirname(current_dir)
        
        # Verify if the provided file path is absolute using os.path.isabs
        if not os.path.isabs(file_path):
            # Resolve relative paths by prefixing them with the calculated project root directory path
            resolved_path = os.path.join(project_root, file_path)
        else:
            # Use the absolute path directly as provided by the caller
            resolved_path = file_path
            
        # Check if the resolved file system path points to an actual file on disk to prevent process spawning failures
        if not os.path.exists(resolved_path):
            # Print a diagnostic warning message to the standard error stream detailing the missing audio resource
            sys.stderr.write(f"[Audio Playback] Error: Audio file not found at path: {resolved_path}\n")
            # Flush the standard error stream buffer to output the error string immediately to the console
            sys.stderr.flush()
            # Return from the function early to prevent executing a subprocess command with a non-existent path
            return
            
        # Determine whether to execute the playback command synchronously or asynchronously based on wait flag
        if wait:
            # Execute the 'afplay' command synchronously using subprocess.run, blocking the calling thread until it finishes
            subprocess.run(["afplay", resolved_path], check=True)
        else:
            # Fork a background process via subprocess.Popen to run 'afplay' asynchronously, letting the Python process continue
            subprocess.Popen(["afplay", resolved_path])
            
    # Capture any OS-level errors, subprocess spawn exceptions, or path resolution failures
    except Exception as e:
        # Output the exception details to the standard error stream for developer diagnostics and logging
        sys.stderr.write(f"[Audio Playback] Error playing audio: {e}\n")
        # Flush the standard error output buffer to ensure immediate visibility of the warning message
        sys.stderr.flush()
