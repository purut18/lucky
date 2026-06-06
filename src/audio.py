"""
This module handles audio recording from the system's microphone and captures keyboard events to support push-to-talk functionality.
It listens for the Alt/Option key to start and stop audio recording, and records the raw input into standard format bytes.

Technical Details:
- Uses the pynput library to monitor global key presses and releases of the Alt key.
- Utilizes threading.Event signals to synchronize key listener events with the main thread.
- Employs sounddevice.RawInputStream to capture mono, 16-bit PCM integer audio stream data at a given sample rate (e.g., 16000Hz).
- Operates a callback-driven buffer queue to record chunked mic audio into memory safely without audio dropouts.
"""

# Import the sys library to output messages and flush text directly to the console output
import sys

# Import the threading module to support multi-threaded events and thread-safe signaling
import threading

# Import the sounddevice library to interact with external/internal sound cards and microphones
import sounddevice as sd

# Import the keyboard module from pynput to capture real-time keyboard inputs outside the console window
from pynput import keyboard

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
