import torch as T
from torch_utils import verify_device

# Assuming you have your agent code imported
from rl_agents import DDPG
from env_wrapper import GymnasiumWrapper
from models import ActorModel, CriticModel

# Create a simple test script to verify the device
if __name__ == "__main__":
    # Create a simple environment
    env = GymnasiumWrapper("BipedalWalker-v3")
    
    # Create a simple actor and critic model
    actor_model = ActorModel(
        env=env,
        layer_config=[
            {'type': 'dense', 'params': {'units': 400, 'kernel': 'variance_scaling', 'kernel params': {'scale': 1.0, 'mode': 'fan_in', 'distribution': 'uniform'}}},
            {'type': 'relu'},
            {'type': 'dense', 'params': {'units': 300, 'kernel': 'variance_scaling', 'kernel params': {'scale': 1.0, 'mode': 'fan_in', 'distribution': 'uniform'}}},
            {'type': 'relu'}
        ],
        device='cuda'
    )
    
    critic_model = CriticModel(
        env=env,
        state_layers=[
            {'type': 'dense', 'params': {'units': 400, 'kernel': 'variance_scaling', 'kernel params': {'scale': 1.0, 'mode': 'fan_in', 'distribution': 'uniform'}}},
            {'type': 'relu'}
        ],
        merged_layers=[
            {'type': 'dense', 'params': {'units': 300, 'kernel': 'variance_scaling', 'kernel params': {'scale': 1.0, 'mode': 'fan_in', 'distribution': 'uniform'}}},
            {'type': 'relu'}
        ],
        device='cuda'
    )
    
    # Create a DDPG agent
    original_agent = DDPG(
        env=env,
        actor_model=actor_model,
        critic_model=critic_model,
        device='cuda'
    )
    
    # Clone the agent and move to CPU
    cpu_agent = original_agent.clone(device='cpu')
    
    # First check if the original agent is correctly on CUDA
    print("\n=== ORIGINAL AGENT (CUDA) VERIFICATION ===")
    stats = verify_device(original_agent, 'cuda', verbose=False)
    print(f"Total components checked: {stats['total']}")
    print(f"Components on correct device (cuda): {stats['correct']}")
    print(f"Components on incorrect device: {stats['incorrect']}")
    if stats['incorrect'] > 0:
        print(f"Incorrect devices found: {stats['incorrect_devices']}")
    
    # Then check if the cloned agent is correctly on CPU
    print("\n=== CLONED AGENT (CPU) VERIFICATION ===")
    stats = verify_device(cpu_agent, 'cpu', verbose=False)
    print(f"Total components checked: {stats['total']}")
    print(f"Components on correct device (cpu): {stats['correct']}")
    print(f"Components on incorrect device: {stats['incorrect']}")
    if stats['incorrect'] > 0:
        print(f"Incorrect devices found: {stats['incorrect_devices']}")
        
    # If there are incorrect devices, run a more verbose check to identify them
    if stats['incorrect'] > 0:
        print("\n=== DETAILED CHECK OF PROBLEMATIC COMPONENTS ===")
        verify_device(cpu_agent, 'cpu', verbose=True) 