"""
================================================================================
Lucky Core Engine - Translation Bridge Validation Test Suite
================================================================================
This script acts as the verification test harness for the JSON-to-osascript translation
bridge (src/bridge.py). It declares sample mock payloads for every system-level action
defined in the application schema.

Technical Specifications:
- Mock Catalog: Contains representative schemas for all supported application verbs.
- Dynamic Execution: Supports testing the translation of all registered action payloads
  sequentially, or targeting a specific action (e.g. 'system.set_volume') passed via CLI.
- Execution toggle: Operates in dry-run mode (outputting code) by default, and supports
  actual execution on macOS when invoked with the '--execute' command-line flag.
"""

# Import the sys module to parse command line parameters and write messages
import sys
# Import the json utility to serialize the mock action payloads for parser input
import json

# Import the orchestrator bridge and executor interfaces from package namespaces
from src import bridge
from src import executor

# Define standard mock action payloads covering all actions in actions.py
MOCK_ACTIONS = {
    "finder.open": {
        "app": "finder",
        "action": "open",
        "params": {}
    },
    "finder.close": {
        "app": "finder",
        "action": "close",
        "params": {"scope": "current"}
    },
    "finder.empty_trash": {
        "app": "finder",
        "action": "empty_trash",
        "params": {}
    },
    "finder.get_active_window_path": {
        "app": "finder",
        "action": "get_active_window_path",
        "params": {}
    },
    "finder.create_folder": {
        "app": "finder",
        "action": "create_folder",
        "params": {"location": "~/Desktop", "name": "LuckyFolderTest"}
    },
    "system.set_volume": {
        "app": "system",
        "action": "set_volume",
        "params": {"level": 50}
    },
    "system.update_volume": {
        "app": "system",
        "action": "update_volume",
        "params": {"amount": 10}
    },
    "system.set_brightness": {
        "app": "system",
        "action": "set_brightness",
        "params": {"level": 80}
    },
    "system.update_brightness": {
        "app": "system",
        "action": "update_brightness",
        "params": {"amount": -15}
    },
    "system.read_volume": {
        "app": "system",
        "action": "read_volume",
        "params": {}
    },
    "system.hide_apps": {
        "app": "system",
        "action": "hide_apps",
        "params": {"target": "others"}
    },
    "system.eject_disks": {
        "app": "system",
        "action": "eject_disks",
        "params": {}
    },
    "system.toggle_dark_mode": {
        "app": "system",
        "action": "toggle_dark_mode",
        "params": {"state": "on"}
    },
    "system.sleep": {
        "app": "system",
        "action": "sleep",
        "params": {}
    },
    "system.shutdown": {
        "app": "system",
        "action": "shutdown",
        "params": {}
    },
    "system.restart": {
        "app": "system",
        "action": "restart",
        "params": {}
    },
    "system.lock_screen": {
        "app": "system",
        "action": "lock_screen",
        "params": {}
    },
    # Dictionary entry for validating the System toggle wifi endpoint.
    # Defines the app name, the routing action string, and mock parameter state.
    "system.toggle_wifi": {
        # Target application module name representing system utilities.
        "app": "system",
        # Specific routing endpoint identification key matching the action verb.
        "action": "toggle_wifi",
        # Structured parameter block to test enabling wifi.
        "params": {
            # Target power state passed to verify the activation command scripting.
            "state": "on"
        }
    },
    "safari.open": {
        "app": "safari",
        "action": "open",
        "params": {}
    },
    "safari.close": {
        "app": "safari",
        "action": "close",
        "params": {"target": "current"}
    },
    "safari.new_tab": {
        "app": "safari",
        "action": "new_tab",
        "params": {"url": "https://www.google.com"}
    },
    "safari.close_tab": {
        "app": "safari",
        "action": "close_tab",
        "params": {"target": "current"}
    },
    "safari.get_active_tab_url": {
        "app": "safari",
        "action": "get_active_tab_url",
        "params": {}
    },
    "safari.get_active_tab_title": {
        "app": "safari",
        "action": "get_active_tab_title",
        "params": {}
    },
    "safari.navigate_tab": {
        "app": "safari",
        "action": "navigate_tab",
        "params": {"direction": "back"}
    },
    "safari.refresh": {
        "app": "safari",
        "action": "refresh",
        "params": {}
    },
    "google chrome.open": {
        "app": "google chrome",
        "action": "open",
        "params": {}
    },
    "google chrome.close": {
        "app": "google chrome",
        "action": "close",
        "params": {"target": "current"}
    },
    "google chrome.new_tab": {
        "app": "google chrome",
        "action": "new_tab",
        "params": {"url": "https://www.github.com"}
    },
    "google chrome.close_tab": {
        "app": "google chrome",
        "action": "close_tab",
        "params": {"target": "current"}
    },
    "google chrome.get_active_tab_url": {
        "app": "google chrome",
        "action": "get_active_tab_url",
        "params": {}
    },
    "google chrome.navigate_tab": {
        "app": "google chrome",
        "action": "navigate_tab",
        "params": {"direction": "forward"}
    },
    "google chrome.refresh": {
        "app": "google chrome",
        "action": "refresh",
        "params": {}
    },
    # Dictionary entry for the Chrome open new window test case.
    # Defines the app name, the routing action string, and mock parameters.
    # We pass a test URL to verify the new window URL translation logic.
    "google chrome.open_new_window": {
        # Specifies the targeted browser application name.
        "app": "google chrome",
        # Specifies the unique verb identification key for spawning windows.
        "action": "open_new_window",
        # Parameter block mapping the parameters schema (includes target navigation url).
        "params": {
            # Target URL string passed to verify setting the browser tab location.
            "url": "https://www.google.com"
        }
    },
    "apple-music.open": {
        "app": "apple-music",
        "action": "open",
        "params": {}
    },
    "apple-music.close": {
        "app": "apple-music",
        "action": "close",
        "params": {}
    },
    "apple-music.change_track": {
        "app": "apple-music",
        "action": "change_track",
        "params": {"direction": "next"}
    },
    "apple-music.toggle_playback": {
        "app": "apple-music",
        "action": "toggle_playback",
        "params": {"state": "play"}
    },
    "apple-music.get_playing_song_details": {
        "app": "apple-music",
        "action": "get_playing_song_details",
        "params": {}
    },
    "apple-music.add_to_favorites": {
        "app": "apple-music",
        "action": "add_to_favorites",
        "params": {}
    },
    "apple-music.remove_from_favorites": {
        "app": "apple-music",
        "action": "remove_from_favorites",
        "params": {}
    },
    "apple-music.shuffle": {
        "app": "apple-music",
        "action": "shuffle",
        "params": {"state": "on"}
    },
    "apple-music.repeat": {
        "app": "apple-music",
        "action": "repeat",
        "params": {"state": "one"}
    },
    "spotify.open": {
        "app": "spotify",
        "action": "open",
        "params": {}
    },
    "spotify.close": {
        "app": "spotify",
        "action": "close",
        "params": {}
    },
    "spotify.change_track": {
        "app": "spotify",
        "action": "change_track",
        "params": {"direction": "previous"}
    },
    "spotify.toggle_playback": {
        "app": "spotify",
        "action": "toggle_playback",
        "params": {"state": "pause"}
    },
    "spotify.get_playing_song_details": {
        "app": "spotify",
        "action": "get_playing_song_details",
        "params": {}
    },
    "spotify.shuffle": {
        "app": "spotify",
        "action": "shuffle",
        "params": {"state": "off"}
    },
    "spotify.repeat": {
        "app": "spotify",
        "action": "repeat",
        "params": {"state": "queue"}
    },
    "spotify.search_and_play": {
        "app": "spotify",
        "action": "search_and_play",
        "params": {"query": "Blinding Lights", "scope": "song"}
    },
    "notes.open": {
        "app": "notes",
        "action": "open",
        "params": {}
    },
    "notes.close": {
        "app": "notes",
        "action": "close",
        "params": {}
    },
    "notes.get_notes": {
        "app": "notes",
        "action": "get_notes",
        "params": {}
    },
    "notes.create_note": {
        "app": "notes",
        "action": "create_note",
        "params": {"title": "Lucky Note Test", "content": "Lucky note content validation line."}
    },
    "notes.get_note_content": {
        "app": "notes",
        "action": "get_note_content",
        "params": {"title": "Lucky Note Test"}
    },
    "notes.update_note": {
        "app": "notes",
        "action": "update_note",
        "params": {"title": "Lucky Note Test", "content": "Additional details."}
    },
    "calendar.open": {
        "app": "calendar",
        "action": "open",
        "params": {}
    },
    "calendar.close": {
        "app": "calendar",
        "action": "close",
        "params": {}
    },
    "calendar.get_events": {
        "app": "calendar",
        "action": "get_events",
        "params": {"date": "2026-06-06"}
    },
    "calendar.create_event": {
        "app": "calendar",
        "action": "create_event",
        "params": {"title": "Project Sprint Meeting", "date": "2026-06-06", "time": "14:30", "duration_in_minutes": 60, "location": "Workspace", "description": "Discuss sprint layout."}
    },
    "calendar.update_event": {
        "app": "calendar",
        "action": "update_event",
        "params": {"title": "Project Sprint Meeting", "date": "2026-06-07", "duration": "90 minutes"}
    },
    "calendar.read_event": {
        "app": "calendar",
        "action": "read_event",
        "params": {"title": "Project Sprint Meeting", "date": "2026-06-07"}
    },
    "reminders.open": {
        "app": "reminders",
        "action": "open",
        "params": {}
    },
    "reminders.close": {
        "app": "reminders",
        "action": "close",
        "params": {}
    },
    "reminders.get_reminders": {
        "app": "reminders",
        "action": "get_reminders",
        "params": {}
    },
    "reminders.create_reminder": {
        "app": "reminders",
        "action": "create_reminder",
        "params": {"title": "Water Plants Reminder", "date": "2026-06-06", "time": "09:00", "notes": "Add liquid plant feed."}
    },
    "reminders.update_reminder": {
        "app": "reminders",
        "action": "update_reminder",
        "params": {"title": "Water Plants Reminder", "new_title": "Water Balcony Plants", "new_notes": "Balcony plants only."}
    },
    "reminders.delete_reminder": {
        "app": "reminders",
        "action": "delete_reminder",
        "params": {"title": "Water Balcony Plants"}
    },
    "textedit.open": {
        "app": "textedit",
        "action": "open",
        "params": {}
    },
    "textedit.close": {
        "app": "textedit",
        "action": "close",
        "params": {}
    },
    "textedit.new_document": {
        "app": "textedit",
        "action": "new_document",
        "params": {}
    },
    "textedit.open_document": {
        "app": "textedit",
        "action": "open_document",
        "params": {"path": "~/Desktop/lucky_test_doc.txt"}
    },
    "router.do-nothing": {
        "app": "router",
        "action": "do-nothing",
        "params": {}
    },
    "router.llm-agent": {
        "app": "router",
        "action": "llm-agent",
        "params": {"task": "Write an email draft replying to Alice"}
    }
}

def run_test(action_key: str, execute_command: bool = False):
    """
    Executes translation verification on a specific registered action model.
    
    Technical Details:
    - Queries MOCK_ACTIONS using the provided key.
    - Serializes the dictionary payload to JSON, then parses it via bridge.action_from_json.
    - Executes translation and outputs the resulting AppleScript/Shell script.
    - If execute_command is True, triggers subprocess execution on the host macOS.
    """
    # Verify that the requested action key is registered in our mock catalog
    if action_key not in MOCK_ACTIONS:
        print(f"Error: Unknown action '{action_key}'. Available options are:")
        for k in sorted(MOCK_ACTIONS.keys()):
            print(f"  {k}")
        return False
        
    # Inform user of test target properties
    print("=" * 80)
    print(f"TESTING ACTION: {action_key}")
    print("=" * 80)
    
    # Retrieve the mock dictionary structure
    payload = MOCK_ACTIONS[action_key]
    # Convert it to a minimized JSON string to replicate model outputs
    json_str = json.dumps(payload)
    print(f"Input JSON: {json_str}\n")
    
    try:
        # Parse and validate the JSON using the bridge's Pydantic deserialization
        action_obj = bridge.action_from_json(json_str)
        # Generate the command code
        cmd = bridge.translate_action_to_command(action_obj)
        
        # Output generated script or notice if bypassed/skipped
        if cmd:
            print("Generated Translation Command:")
            print("-" * 60)
            print(cmd)
            print("-" * 60)
            
            # Execute command if requested
            if execute_command:
                print("Running command on system...")
                exit_code = executor.execute_osascript(cmd)
                print(f"Execution finished. Exit Code: {exit_code}")
        else:
            print("Action parsed successfully but translator returned None (skipped by bridge).")
            
        print("Status: ✓ PASS\n")
        return True
        
    except Exception as err:
        print(f"Status: ✗ FAIL (Error: {err})\n")
        return False

def main():
    """
    Main orchestration routine.
    
    Technical Details:
    - Inspects command line arguments (sys.argv).
    - If '--execute' is present, sets the execution flag and strips it from arguments.
    - If an action key remains in the arguments, validates that specific action.
    - Otherwise, loops through and validates all test cases in sequence.
    """
    # Check if execute flag was provided anywhere in the execution arguments list
    execute_command = False
    args = sys.argv[1:]
    if "--execute" in args:
        execute_command = True
        args.remove("--execute")
        
    # Check if a specific action filter key was supplied by the user
    if args:
        target_action = args[0]
        success = run_test(target_action, execute_command)
        sys.exit(0 if success else 1)
        
    # Default path: execute validation across all entries in the mock database
    print("Running bridge validation tests for all actions (dry-run mode)...\n")
    pass_count = 0
    fail_count = 0
    
    # Loop over mock keys in alphabetical order
    for key in sorted(MOCK_ACTIONS.keys()):
        # Perform validation checks
        success = run_test(key, execute_command)
        if success:
            pass_count += 1
        else:
            fail_count += 1
            
    # Output complete metrics summary
    print("=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Total Tests Run: {pass_count + fail_count}")
    print(f"Passed: {pass_count}")
    print(f"Failed: {fail_count}")
    print("=" * 80)
    
    # Set final process exit code depending on failure states
    sys.exit(0 if fail_count == 0 else 1)

if __name__ == "__main__":
    main()
