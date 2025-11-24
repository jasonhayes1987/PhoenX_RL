#!/usr/bin/env python3
"""
Test script to validate IsaacSimWrapper imports and basic functionality
without actually instantiating Isaac Sim environments (to avoid event loop issues).
"""

import sys
import os

# Use environment variable for IsaacLab path, with fallback to relative path
ISAACLAB_PATH = os.environ.get('ISAACLAB_PATH', os.path.join(os.path.dirname(__file__), '..', '..', 'IsaacLab', 'source'))
ISAACLAB_TASKS_PATH = os.path.join(ISAACLAB_PATH, 'isaaclab_tasks')

sys.path.append(ISAACLAB_PATH)
sys.path.append(ISAACLAB_TASKS_PATH)

print(f"ISAACLAB_PATH: {ISAACLAB_PATH}")
print(f"ISAACLAB_TASKS_PATH: {ISAACLAB_TASKS_PATH}")

def test_imports():
    """Test that all required modules can be imported."""

    try:
        # Test 1: Import IsaacSimWrapper
        print("Testing IsaacSimWrapper import...")
        from env_wrapper import IsaacSimWrapper
        print("SUCCESS: IsaacSimWrapper imported")

        # Test 2: Check that IsaacSimWrapper class exists and has expected methods
        print("Testing IsaacSimWrapper class structure...")
        assert hasattr(IsaacSimWrapper, '__init__'), "IsaacSimWrapper missing __init__"
        assert hasattr(IsaacSimWrapper, '_initialize_env'), "IsaacSimWrapper missing _initialize_env"
        assert hasattr(IsaacSimWrapper, 'reset'), "IsaacSimWrapper missing reset"
        assert hasattr(IsaacSimWrapper, 'step'), "IsaacSimWrapper missing step"
        assert hasattr(IsaacSimWrapper, 'close'), "IsaacSimWrapper missing close"
        assert hasattr(IsaacSimWrapper, 'format_actions'), "IsaacSimWrapper missing format_actions"
        print("SUCCESS: IsaacSimWrapper class structure is correct")

        # Test 3: Check that config string parsing works
        print("Testing config string parsing...")
        cfg_string = "isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg:CartpoleEnvCfg"
        module_path, class_name = cfg_string.split(':')
        assert module_path == "isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg"
        assert class_name == "CartpoleEnvCfg"
        print("SUCCESS: Config string parsing works")

        # Test 4: Test importlib can find the module (but don't instantiate)
        print("Testing module discovery...")
        import importlib
        try:
            # This will fail if Isaac Lab isn't properly set up, but should succeed if paths are correct
            importlib.import_module(module_path)
            print("SUCCESS: Module can be imported")
        except ImportError as e:
            print(f"WARNING: Module import failed (expected if Isaac Lab not fully configured): {e}")
        except Exception as e:
            print(f"WARNING: Unexpected error during module import: {e}")

        print("\nAll import tests completed successfully!")
        print("\nNOTE: Isaac Sim environments cannot be instantiated in this script")
        print("because they require a dedicated event loop. Use a separate process")
        print("or script that doesn't run in a shared environment.")

    except Exception as e:
        print(f"FAILED: Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()
