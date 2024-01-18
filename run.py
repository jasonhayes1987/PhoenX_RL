# Imports and Class Definitions
# (Include all necessary imports and class definitions here, such as Reinforce_Model, Agent, and REINFORCE)
from reinforce import REINFORCE

# wandb sweep configuration
sweep_configuration = {
    "method": "bayes",
    "name": "CartPole-v0_REINFORCE",
    "metric": {"goal": "maximize", "name": "reward"},
    "parameters": {
        "hidden_layer_sizes": {"values": [(10,), (10, 10), (100,),(100, 10), (100, 100)]},
        "num_episodes": {"value": 500},
        "learning_rate": {"max": 0.1, "min": 0.001},
        "gamma": {"max": 1.0, "min": 0.1},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 100,
        "eta": 2,
    },
}

# model training configuration
config = {
    "env": "CartPole-v0",
    "learning_rate": 0.01,
    "gamma": 0.99,
    "num_episodes": 500,
    "hidden_layer_sizes": (10,),
    "project_name": "CartPole-v0_REINFORCE",
    "log_dir": "logs",
    "save_dir": "models",
    "num_sweeps": 0
}

if __name__ == "__main__":
    # Initialize and Train the Agent
    reinforce_agent = REINFORCE(config=config)
    reinforce_agent.train()

    # Optionally, you can add command-line argument parsing here to make the script more flexible
