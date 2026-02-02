#!/usr/bin/env python3
"""
Isaac Lab Setup Test Script
Tests if Isaac Lab and related dependencies are properly installed and configured.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /workspaces/PhoenX_RL
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Current sys.path (first few entries):")
for path in sys.path[:10]:
    print(f"  - {path}")
print()

def test_imports():
    """Test various import scenarios and provide helpful feedback."""
    print("=== Isaac Lab Setup Test ===\n")

    # Test 1: PyTorch (core dependency)
    try:
        import torch as th
        print(f"✓ PyTorch: Version {th.__version__}, CUDA: {th.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch failed: {e}")
        return False

    # Test 2: Isaac Lab core
    try:
        import isaaclab
        print(f"✓ Isaac Lab core: {isaaclab.__version__} at {isaaclab.__file__}")
    except ImportError as e:
        print(f"✗ Isaac Lab core failed: {e}")
        print("  Fix: Run 'pip install -e /IsaacLab/source/isaaclab' or add to PYTHONPATH")
        return False

    # Test 3: Isaac Lab envs/tasks (these depend on Isaac Sim, so handle gracefully)
    isaac_sim_available = False
    try:
        # Check for Isaac Sim logging (key dependency)
        import omni.log
        isaac_sim_available = True
        print("✓ Isaac Sim logging (omni.log) available")
    except ImportError as e:
        print(f"⚠ Isaac Sim logging not available: {e}")
        print("  This is common in headless/Docker setups. Isaac Sim requires full runtime (carb/Omniverse).")

    if isaac_sim_available:
        try:
            from isaaclab.envs import DirectRLEnv
            from isaaclab_tasks.direct.cartpole import CartpoleEnvCfg
            print("✓ Isaac Lab envs/tasks imported successfully (DirectRLEnv, CartpoleEnvCfg)")
            
            # Test config instantiation (no sim launch needed)
            cfg = CartpoleEnvCfg(num_envs=1, sim_device="cpu")
            print("✓ Cartpole config instantiated successfully.")
        except ImportError as e:
            print(f"✗ Isaac Lab envs/tasks failed (even with Isaac Sim): {e}")
            return False
    else:
        print("⚠ Skipping envs/tasks test (requires Isaac Sim). Install full Isaac Sim or run in Omniverse environment.")

    print("\n=== Setup Check Complete ===")
    print("Your base setup (PyTorch + isaaclab) looks good! For full simulation, ensure Isaac Sim is sourced.")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)