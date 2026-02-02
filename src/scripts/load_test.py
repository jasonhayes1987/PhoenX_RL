import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from app.agent_utils import load_agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DDPG Agent from config file')
    parser.add_argument('--agent_dir', type=str, required=True, help='Path to the agent configuration directory')
    args = parser.parse_args()
    agent = load_agent(args.agent_dir)
    print(agent.get_config())