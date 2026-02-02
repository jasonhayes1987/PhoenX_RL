import sys
sys.path.append('/workspace/isaaclab/source')
sys.path.append('/workspace/isaaclab/source/isaaclab_tasks')

from isaaclab.app import AppLauncher

# Launch Isaac Sim headless to load extensions
app_launcher = AppLauncher(headless=True, device="cuda")
simulation_app = app_launcher.app

def test_gpu_physx():
    try:
        from omni.physx import acquire_physx_interface
        physx = acquire_physx_interface()
        print('GPU Supported:', physx.get_gpu_found())
        # Note: is_gpu_pipeline may not be available; check logs for PhysX GPU status
        print('PhysX interface acquired successfully')
    except Exception as e:
        print('Error accessing PhysX:', str(e))

if __name__ == "__main__":
    test_gpu_physx()
    simulation_app.close()  # Clean up