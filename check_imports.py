#!/usr/bin/env python3
"""
Quick diagnostics script to check for import errors and basic functionality.
"""

import sys
import os

# Unset the shell environment variable
if 'LLM_MODEL_NAME' in os.environ:
    del os.environ['LLM_MODEL_NAME']

print("Testing imports...")
print("-" * 60)

try:
    print("1. Importing data_model...", end=" ")
    from data_model import LANGUAGE_SELECTION, COLLECTING_HISTORY, CRISIS_MODE
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("2. Importing db...", end=" ")
    import db
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("3. Importing llm...", end=" ")
    import llm
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("4. Importing utils...", end=" ")
    import utils
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("5. Importing safety...", end=" ")
    import safety
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("6. Importing progress...", end=" ")
    import progress
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("7. Importing transitions...", end=" ")
    import transitions
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("8. Importing report...", end=" ")
    import report
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("9. Importing handlers...", end=" ")
    from handlers import (
        start_handler,
        language_selected,
        handle_patient_message,
        pause_session,
        resume_session,
        report_handler,
        cancel_handler,
        help_handler,
    )
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("-" * 60)
print("✓ All imports successful!")
print()
print("Bot modules are ready. You can now run: python3 bot.py")
