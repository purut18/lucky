"""
This module initializes the 'src' directory as a Python package.
By marking this directory as a package, we allow other Python files (like our main 'transcribe.py' at the root)
to import files from inside this folder (like 'from src.config import load_configuration').
Technical details:
- Allows namespace packaging and absolute/relative imports within the module.
- We expose the main modules so they are directly accessible when importing the package if desired.
"""

# Let's import config module so other scripts can access config details directly from the package
from . import config

# Let's import audio module so that other scripts can access the audio recording services
from . import audio

# Let's import transcriber module so that speech-to-text systems can be instantiated easily
from . import transcriber

# Let's import translator module so that text translation code works out-of-the-box
from . import translator

# Let's import executor module so that commands can be run seamlessly on the system terminal
from . import executor

# Let's import bridge module so that Pydantic actions can be translated to executable scripts
from . import bridge
