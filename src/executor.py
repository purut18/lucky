"""
This module is responsible for executing system-level AppleScript (osascript) commands on the host macOS.
It runs the commands in a separate sub-process and prints the output (stdout) or errors (stderr) to the console.

Technical Details:
- Uses the built-in python 'subprocess' library.
- Runs via 'subprocess.run' with 'shell=True' so that the shell parses the complex piping/formatting (e.g., EOF blocks, AppleScript strings).
- Enables 'text=True' to receive command output as standard string text rather than raw byte sequences.
- Captures stdout and stderr into memory buffers ('capture_output=True') to avoid cluttering the parent process, then formats them for readable printing.
"""

# Import the system utility 'sys' to write and flush output directly to the terminal interface
import sys

# Import the standard 'subprocess' module to allow running external shell commands on the computer
import subprocess

def execute_osascript(osascript_code):
    """
    Executes a string of AppleScript code inside a macOS terminal shell.
    Captures and prints the standard output, standard error, and exit codes.
    """
    # Verify that the generated AppleScript code string is not empty before running it
    if osascript_code:
        # Protect our execution step with a try-except block to handle run-time system errors gracefully
        try:
            # Inform the user on the console screen that the script is currently starting execution
            sys.stdout.write("[Running on terminal...]\n")
            # Force the system to print out the message immediately without waiting in buffer queues
            sys.stdout.flush()
            # Run the command code inside the shell, capturing outputs as text characters instead of binary data
            result = subprocess.run(osascript_code, shell=True, text=True, capture_output=True)
            # If the command outputted any success information to stdout, print it to the console
            if result.stdout:
                # Write the standard output text lines to our terminal console screen
                sys.stdout.write(f"Stdout:\n{result.stdout}\n")
            # If the command outputted any warning or error information to stderr, print it to the console
            if result.stderr:
                # Write the standard error text lines to our terminal console screen
                sys.stdout.write(f"Stderr:\n{result.stderr}\n")
            # Print a closing statement showing the execution is complete and the final system status code
            sys.stdout.write(f"[Finished execution with return code {result.returncode}]\n\n")
            # Flush the output stream to ensure the status message shows up on the screen immediately
            sys.stdout.flush()
            # Return the process exit status number (typically 0 means success, others mean errors)
            return result.returncode
        # Catch any system-level execution errors (like command not found or invalid permissions)
        except Exception as exec_err:
            # Print the detailed failure error message to inform the user why it did not run
            sys.stdout.write(f"Execution failed: {exec_err}\n\n")
            # Flush the console output stream to display the error text instantly
            sys.stdout.flush()
            # Return a status code of -1 to represent an internal execution exception occurred
            return -1
    # If the provided script text was empty, return 0 representing no work was needed
    return 0
