"""
================================================================================
Lucky Core Engine - Background LLM Agent Module (agent.py)
================================================================================
This module implements a non-blocking background LLM research agent utility.
It listens for 'router.llm-agent' actions delegated by the main orchestrator,
spawns parallel threads to process incoming user requests, and queries the
OpenAI Responses API with background execution enabled (background=True).
To ensure durability, it stores and tracks all task execution states inside
a local JSON database ('localDB/agents.json'). If the application is closed
or net connectivity is lost, pending background tasks are automatically
resumed and completed upon the next application startup.

Technical Specifications:
- Multithreading: Uses the built-in python 'threading' library to execute non-blocking,
  concurrent daemon threads, allowing multiple independent background tasks to run.
- Thread Safety: Implements a synchronization Lock (db_lock) to prevent concurrent write
  race conditions on the local tracking database file.
- Responses API: Integrates 'client.responses.parse' for structured, validated JSON responses,
  leveraging agentic tool sets like 'web_search'.
- Durable Tracking: Keeps a local registry of response IDs, status ('pending', 'completed', 'failed'),
  prompts, and created timestamps.
- Polling Loop: Periodically queries response status until resolved, then saves the finalized
  output as a Markdown file in '~/Documents/Lucky/'.
"""

# Import the 'os' library to perform secure file system operations and locate home directories
import os
# Import 'sys' to write and flush messages directly to standard output and error stream buffers
import sys
# Import 'time' to capture floating-point epoch timestamps for measuring process durations and timing wait loops
import time
# Import 'json' to parse and serialize tracking records to the local JSON database file
import json
# Import 'threading' to run concurrent operations in parallel and initialize synchronization locks
import threading
# Import 'BaseModel' and 'Field' from Pydantic to build structured JSON schema validation layers
from pydantic import BaseModel, Field
# Import 'OpenAI' from the official openai package to communicate with Responses API endpoints
from openai import OpenAI
# Import our application-wide configurations module to read keys, model settings, and prompts
from src import config
# Import the global audio utility module to enable playing notification sounds throughout the application
# This module provides a cross-platform function `play_audio` that utilizes the native macOS 'afplay' command
# to asynchronously or synchronously play audio files located relative to the project root.
from src import audio

# Define the ANSI escape sequence for changing the terminal font foreground color to green
ANSI_GREEN = "\033[92m"
# Define the ANSI escape sequence for changing the terminal font foreground color to red
ANSI_RED = "\033[91m"
# Define the ANSI escape sequence to reset terminal font styles and colors to default
ANSI_RESET = "\033[0m"

# Compute the absolute directory path of the active project root dynamically at runtime.
# os.path.abspath resolves any relative dots or links into a fully expanded, canonical path structure.
# __file__ is a Python magic global variable containing the absolute or relative path of this specific script file ('src/agent.py').
# The first call to os.path.dirname retrieves the container directory of this file, which resolves to the 'src' directory.
# The second nested call to os.path.dirname ascends another level to retrieve the root directory of the repository/project.
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the absolute target path pointing to the local JSON database file inside the root's localDB folder.
# os.path.join handles file system separators cross-platform (e.g. Unix-based '/' and Windows-based '\') dynamically.
# This ensures that the application remains fully portable, working seamlessly across different users and execution environments.
DATABASE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "localDB", "agents.json")

# Synchronization lock object to guarantee mutually exclusive access to the local JSON file
db_lock = threading.Lock()


class AgentResultSchema(BaseModel):
    """
    Pydantic Schema defining the required structure for the LLM agent's output.
    Ensures that the model returns both a valid destination filename and markdown content.
    """
    # Enforce a non-empty string for the target filename, excluding paths or extensions
    file_name: str = Field(
        description="The desired name of the generated file (e.g., 'WeeklyReport' or 'Ideas'), without spaces or extension."
    )
    # Enforce a string containing the complete markdown body containing the researched response
    markdown_content: str = Field(
        description="The complete and detailed markdown text answering the user's questions."
    )


def _load_database() -> dict:
    """
    Reads the local tracking database from the JSON file.
    If the file does not exist, it creates the folder and the file with an empty JSON object.
    If the file fails to parse, returns an empty dictionary.
    
    Returns:
        dict: The mapping of response IDs to their metadata records.
    """
    # Verify if the database file is physically present in the local file system.
    # os.path.exists performs a low-level system call to verify the existence of the path.
    if not os.path.exists(DATABASE_FILE_PATH):
        # Extract the directory portion of the absolute database path.
        # os.path.dirname resolves the container directory (e.g. 'localDB') for creating it recursively.
        db_dir = os.path.dirname(DATABASE_FILE_PATH)
        # Verify if the target parent folder exists in the physical file system.
        if not os.path.exists(db_dir):
            try:
                # Recursively generate directories along the path.
                # os.makedirs raises OSError if permissions are lacking or paths are blocked.
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                # Write errors to the system standard error stream to notify the user of filesystem errors.
                sys.stderr.write(f"[LLM Agent DB] Error creating directory {db_dir}: {e}\n")
                # Flush the stream buffer to render the output in the console immediately.
                sys.stderr.flush()
        try:
            # Open the file in write mode, replacing any existing content with UTF-8 text encoding.
            # Using 'with' block ensures the file descriptor is securely closed even if exceptions occur.
            with open(DATABASE_FILE_PATH, "w", encoding="utf-8") as f:
                # Write an empty JSON object string representation to initialize the local database file.
                # json.dump serializes python dictionary into JSON formatting with clean indentation of 4 spaces.
                json.dump({}, f, indent=4)
        except Exception as e:
            # Write errors to standard error if file creation or serialization fails.
            sys.stderr.write(f"[LLM Agent DB] Error initializing file {DATABASE_FILE_PATH}: {e}\n")
            # Flush stderr to output the error logs immediately.
            sys.stderr.flush()
        # Return an empty dictionary because the newly created file has no transaction history records yet.
        return {}
    # Wrap reading in a try block to gracefully capture file corruption errors or read blockages.
    try:
        # Open the file in read mode enforcing standard utf-8 encoding.
        # Standardize on UTF-8 to prevent character encoding mismatch issues across platforms.
        with open(DATABASE_FILE_PATH, "r", encoding="utf-8") as f:
            # Parse and return the JSON object as a Python dictionary structure.
            # json.load decodes the JSON formatted string from the file stream into python dictionary keys and values.
            return json.load(f)
    # Catch any deserialization, read access, or permission exceptions.
    # This acts as a fallback to ensure application uptime even if database file gets corrupted.
    except Exception:
        # Fall back to returning an empty dictionary to keep background daemon threads running stably.
        return {}


def _save_database(db: dict) -> None:
    """
    Serializes and writes the tracking database dictionary to the local JSON file.
    
    Args:
        db (dict): The target dictionary containing all agent records.
    """
    # Extract the directory path containing our database file
    db_dir = os.path.dirname(DATABASE_FILE_PATH)
    # Check if the target folder exists in the file system
    if not os.path.exists(db_dir):
        # Recursively create the directory and any required parent paths
        os.makedirs(db_dir, exist_ok=True)
    # Wrap writing operations in a try block to handle write permission issues
    try:
        # Open the file in write mode, replacing any existing content
        with open(DATABASE_FILE_PATH, "w", encoding="utf-8") as f:
            # Serialize and write the dict to JSON format with clean indentation
            json.dump(db, f, indent=4)
    # Catch any file system or writing errors
    except Exception as e:
        # Print a descriptive warning to the standard error stream
        sys.stderr.write(f"[LLM Agent DB] Error saving database: {e}\n")
        # Flush standard error to output the message immediately
        sys.stderr.flush()


def update_agent_status(response_id: str, status: str, task: str = "", file_name: str = None) -> None:
    """
    Thread-safe utility to insert or update the state records inside the local JSON database.
    
    Args:
        response_id (str): The unique transaction key returned by OpenAI.
        status (str): The current execution status ('pending', 'completed', 'failed').
        task (str, optional): The prompt description string.
        file_name (str, optional): The resolved sanitized output file name.
    """
    # Acquire the database lock to block other threads from writing simultaneously
    with db_lock:
        # Read the current database contents into memory
        db = _load_database()
        # Check if the target response key is already registered in the registry
        if response_id not in db:
            # Insert a new record dictionary initialized with status and timestamps
            db[response_id] = {
                "task": task,
                "status": status,
                "created_at": time.time(),
                "file_name": file_name
            }
        else:
            # Update the status string of the existing record
            db[response_id]["status"] = status
            # If a file name is provided, update the metadata reference field
            if file_name:
                # Update the target filename in the database dictionary
                db[response_id]["file_name"] = file_name
        # Commit the updated dictionary back to the local storage file
        _save_database(db)


class LlmAgentExecutor:
    """
    Executor class that handles background thread execution, OpenAI Responses API calls,
    polling status loops, database logging, and markdown file creation.
    """

    def __init__(self, task: str, response_id: str = None):
        """
        Constructor method to initialize the executor instance with a task prompt.
        Supports optional response_id instantiation to resume polling existing tasks.
        
        Args:
            task (str): The natural language instruction describing the research target.
            response_id (str, optional): An existing transaction ID to recover and poll.
        """
        # Save the natural language task string to the instance scope
        self.task = task
        # Save the optional response transaction ID to the instance scope
        self.response_id = response_id
        # Extract the OpenAI API token from the config module instance
        self.api_key = config.OPENAI_API_KEY
        # Retrieve the configured Large Language Model name from config variables
        self.model = getattr(config, "LLM_AGENT_MODEL", "gpt-5.4")
        # Retrieve the reasoning effort setting (low/medium/high) from config variables
        self.reasoning_effort = getattr(config, "LLM_AGENT_REASONING_EFFORT", "medium")
        # Retrieve the system prompt instructing the agent on its role and rules from config
        self.system_prompt = getattr(config, "LLM_AGENT_SYSTEM_PROMPT", "You are a helpful assistant.")
        # Retrieve the boolean flag indicating if web search grounding should be utilized
        self.use_internet = getattr(config, "LLM_AGENT_USE_INTERNET", False)

    def execute_flow(self) -> None:
        """
        Runs the complete execution pipeline inside a background thread.
        1. Submits task to Responses API if no response_id exists.
        2. Logs task as 'pending' in the database.
        3. Polls status until 'completed' or 'failed'.
        4. Writes output to Markdown file and updates database status.
        """
        # Capture the system epoch time in seconds at the start of the execution pipeline
        start_time = time.time()
        # Retrieve the current thread's unique integer identifier for log differentiation
        thread_id = threading.get_ident()

        # Wrap execution in a try block to capture API errors, file failures, or network interruptions
        try:
            # Raise an error immediately if the OpenAI API Key is missing from the environment
            if not self.api_key:
                # Raise ValueError detailing missing credentials to be caught by the handler
                raise ValueError("OPENAI_API_KEY is not defined in your environment configuration.")

            # Instantiate a new OpenAI API client instance passing our configured API key
            client = OpenAI(api_key=self.api_key)

            # Step 1: Submit background request if we do not have an existing response_id
            if not self.response_id:
                # Format and output a green-colored log indicating task submission is starting
                sys.stdout.write(f"{ANSI_GREEN}[LLM Agent Thread {thread_id}] Started working... {ANSI_RESET}\n")
                # Immediately after initiating the background LLM agent thread, play a confirmation sound to audibly signal
                # that the background task has been successfully dispatched. The audio is played asynchronously (`wait=False`)
                # so that it does not block the ongoing execution of the thread or subsequent logging.
                audio.play_audio("audio/sure-on-it.mp3", wait=False)
                # Flush stdout to force the terminal to print the log immediately without buffering
                sys.stdout.flush()

                # Initialize an empty list to compile API tool parameters
                tools = []
                # Check if search grounding is requested by the configuration flag
                if self.use_internet:
                    # Append the web search tool schema dictionary to tools
                    tools.append({"type": "web_search"})

                # Execute background submission to responses.parse with background=True
                response = client.responses.parse(
                    # Inject the configured model name string
                    model=self.model,
                    # Supply the task description prompt as input
                    input=self.task,
                    # Supply system prompts instructing the agent
                    instructions=self.system_prompt,
                    # Inject the Pydantic schema model to enforce structured validation formats
                    text_format=AgentResultSchema,
                    # Configure the reasoning effort depth constraints
                    reasoning={"effort": self.reasoning_effort},
                    # Supply the tools parameter if tools are present
                    tools=tools if tools else None,
                    # Instruct the API to run the request asynchronously in the background
                    background=True
                )
                
                # Retrieve the newly generated unique response transaction identifier string
                self.response_id = response.id
                
                # Register the transaction status as 'pending' inside our local tracking database
                update_agent_status(self.response_id, "pending", task=self.task)

                # Output a success log to inform the user the transaction was submitted
                # sys.stdout.write(f"{ANSI_GREEN}[LLM Agent Thread {thread_id}] Task submitted successfully. Response ID: {self.response_id}{ANSI_RESET}\n")
                # Flush stdout to update the user terminal interface immediately
                sys.stdout.flush()
            else:
                # Output a log informing that the thread is resuming polling of a previous task
                # sys.stdout.write(f"{ANSI_GREEN}[LLM Agent Thread {thread_id}] Resuming status polling for: '{self.task}' (ID: {self.response_id})...{ANSI_RESET}\n")
                # Flush stdout to force terminal rendering
                sys.stdout.flush()

            # Step 2: Poll response status periodically until completed, failed, or cancelled
            poll_interval = 3.0  # Polling interval frequency in seconds
            max_retries = 300   # Maximum retry attempts before timing out
            
            # Start the iteration loop representing polling attempts
            for attempt in range(max_retries):
                # Retrieve the current status payload from the OpenAI Responses endpoint
                retrieved = client.responses.retrieve(response_id=self.response_id)
                # Extract the current processing status string from retrieved object
                status = retrieved.status

                # Check if the status has resolved to completed successfully
                if status == "completed":
                    # Initialize an empty variable to hold the extracted JSON result string
                    json_str = None
                    # Iterate through the output items list returned by the API
                    for out_item in retrieved.output:
                        # Find the output item representing the assistant's message response
                        if out_item.type == "message":
                            # Iterate through the message content components
                            for content_item in out_item.content:
                                # Locate the text segment containing the generated payload
                                if content_item.type == "output_text":
                                    # Copy the raw output text content containing the structured JSON
                                    json_str = content_item.text
                                    # Break out of content loop
                                    break
                            # Break out of output item loop if text was resolved
                            if json_str:
                                # Stop processing items
                                break

                    # Raise ValueError if the completed payload did not contain any valid text content
                    if not json_str:
                        # Raise exception to fail execution branch
                        raise ValueError("Completed background response did not contain output text.")

                    # Deserialize and validate the raw JSON string using Pydantic model validation
                    parsed_data = AgentResultSchema.model_validate_json(json_str)
                    # Extract the raw file name from the validated Pydantic object
                    raw_file_name = parsed_data.file_name.strip()
                    # Extract the raw markdown text content from the validated Pydantic object
                    markdown_content = parsed_data.markdown_content

                    # Sanitize the file name to prevent directory traversal attacks by taking only the basename
                    file_name = os.path.basename(raw_file_name)
                    # Remove any trailing .md or .MD extension if the model accidentally included it
                    if file_name.lower().endswith(".md"):
                        # Slices the filename string to strip the last 3 characters
                        file_name = file_name[:-3]
                    # Replace any spaces in the filename with underscores to keep files system-friendly
                    file_name = file_name.replace(" ", "_")

                    # Expand the relative folder string to get the absolute path to the user's Documents folder
                    user_documents_dir = os.path.expanduser("~/Documents")
                    # Construct the absolute path targeting the subfolder Documents/Lucky
                    lucky_dir = os.path.join(user_documents_dir, "Lucky")
                    # Safely create the target directory and any required parent folders recursively
                    os.makedirs(lucky_dir, exist_ok=True)
                    # Join the absolute folder path and the sanitized filename with a trailing .MD extension
                    output_path = os.path.join(lucky_dir, f"{file_name}.MD")

                    # Open the target path in write mode enforcing utf-8 text encoding standard
                    with open(output_path, "w", encoding="utf-8") as f:
                        # Write the complete generated markdown content into the newly created file
                        f.write(markdown_content)

                    # Update the task status to 'completed' and register filename in local JSON database
                    update_agent_status(self.response_id, "completed", file_name=file_name)

                    # Compute the total elapsed time by subtracting start time from the current timestamp
                    duration = time.time() - start_time
                    # Write a success status log in green color to inform that the markdown file is ready
                    sys.stdout.write(f"{ANSI_GREEN}[LLM Agent Thread {thread_id}] done. Created file: {output_path} (Duration: {duration:.2f}s){ANSI_RESET}\n")
                    # Flush stdout to synchronize the success log display in the user's terminal
                    sys.stdout.flush()
                    # End execution thread successfully
                    return

                # Check if the status has resolved to a failed or cancelled terminal state
                elif status in ["failed", "cancelled"]:
                    # Raise a ValueError detailing the failure code returned by OpenAI
                    raise ValueError(f"OpenAI Response processing ended with status: '{status}'.")

                # Sleep the thread execution for the configured interval before the next status poll
                time.sleep(poll_interval)
            # Executed if the loop completes without resolving status
            else:
                # Raise TimeoutError detailing that retry limit was exceeded
                raise TimeoutError("Polling background response exceeded maximum retries.")

        # Catch any exceptions or errors that occurred during execution
        except Exception as err:
            # Compute the total time elapsed before the failure occurred
            duration = time.time() - start_time
            # Write a red-colored error message to the system's standard error stream
            sys.stderr.write(f"{ANSI_RED}[LLM Agent Thread {thread_id}] Error in background execution after {duration:.2f}s: {err}{ANSI_RESET}\n")
            # Flush the error buffer to display the message instantly
            sys.stderr.flush()
            # If a transaction ID exists, log the failure state in the local JSON database
            if self.response_id:
                # Update the target record status to 'failed' in local storage
                update_agent_status(self.response_id, "failed")


def resume_pending_agents() -> None:
    """
    Scans the local JSON database for any background agent tasks that are still 'pending'.
    For each pending task, spawns a new background thread to resume status polling.
    """
    # Load all database records into memory
    db = _load_database()
    # Initialize a counter to track the number of resumed tasks
    pending_count = 0
    
    # Iterate through all entries in the database dict to find pending status keys
    for response_id, data in db.items():
        # Check if the record status field equals pending
        if data.get("status") == "pending":
            # Extract the original prompt task description
            task = data.get("task", "")
            # Instantiate a new LlmAgentExecutor in recovery mode passing the response ID
            executor = LlmAgentExecutor(task=task, response_id=response_id)
            # Instantiate a Thread targeting the execute_flow method of the executor
            bg_thread = threading.Thread(target=executor.execute_flow, daemon=True)
            # Start background execution thread polling
            bg_thread.start()
            # Increment the counter
            pending_count += 1
            
    # If any pending tasks were found and resumed, log a green-colored message
    if pending_count > 0:
        # Output summary details to the console screen
        sys.stdout.write(f"{ANSI_GREEN}[LLM Agent] Resumed polling for {pending_count} pending background tasks.{ANSI_RESET}\n")
        # Flush standard output stream
        sys.stdout.flush()


def start_agent_thread(task: str) -> None:
    """
    Entrypoint function that instantiates the executor and spawns a new parallel daemon thread.
    This function returns immediately to the calling thread without blocking.
    
    Args:
        task (str): The prompt task content passed from the speech translator.
    """
    # Instantiate the LlmAgentExecutor class with the requested task string
    executor = LlmAgentExecutor(task)
    # Instantiate a Thread targeting the execute_flow method of the executor instance
    bg_thread = threading.Thread(target=executor.execute_flow, daemon=True)
    # Activate the thread, starting background operations immediately
    bg_thread.start()
