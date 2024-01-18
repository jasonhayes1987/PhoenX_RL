
# RL_Agents

## Overview
RL_Agents is a comprehensive portfolio of reinforcement learning (RL) agents, designed to demonstrate the implementation and functionality of various RL algorithms. This project provides a practical application of reinforcement learning techniques in complex environments.

## Features
- Implementation of multiple reinforcement learning agents.
- Integration with [Weights & Biases](https://wandb.ai/site) for experiment tracking.
- Support for interactive visualizations using Streamlit.
- Customizable models and training environments.

## File Descriptions
- `rl_agents.py`: Core implementations of the reinforcement learning agents.
- `streamlit_support.py`: Streamlit interface for interactive visualization and control.
- `wandb_support.py`: Integration utilities for experiment tracking with Weights & Biases.
- `app.py`: Main application entry point.
- `helper.py`: Helper functions for various tasks like optimizer selection.
- `models.py`: Deep learning models used by the RL agents.

## Installation
To install and run RL_Agents, follow these steps:
1. Clone the repository: `git clone https://github.com/jasonhayes1987/RL_Agents.git`
2. Navigate to the cloned directory: `cd RL_Agents`
3. (Optional) Create a virtual environment: `python -m venv env`
4. Activate the environment: Windows: `.\env\Scriptsctivate`, Linux/Mac: `source env/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`

## Usage

### Running the Application
1. **Start the Application**: Run `app.py` using Streamlit by executing the command `streamlit run app.py` in your terminal. This launches the main interface of RL_Agents.

### Building Models
2. **Model Building**: In the main menu, navigate to the "Build Model" section. Here, you can create custom models based on your requirements:
   - **Select Model Type**: Choose the type of model you want to build, along with its parameters.
   - **Configure Policy and Value Models**: Customize the policy model and value model configurations according to your needs.
   - **Save Directory**: Specify the directory where you want to save your model.
   - **Build and Save**: Click "Build Model" to construct your model and save it to the specified directory.

### Training Models
3. **Model Training**: Use the "Train Model" section for training your models.
   - **Load a Model**: Load an existing model from a folder or a Weights & Biases (WandB) artifact. If logged in to WandB, select the project and training run for the artifact.
   - **Environment and Episodes**: Choose the gym environment for training and specify the number of training episodes.
   - **Rendering Options**: To visualize the training, enable 'Render Training Episodes' and set the frequency of rendering in 'Render Frequency'. Rendered episodes are saved to `save_directory/renders`.
   - **Start Training**: Initiate the training process by selecting "Start Training".

### Testing Models
4. **Model Testing**: In the "Test Model" section, you can evaluate the performance of your models.
   - **Model Loading**: Similar to training, load a model from a local directory or from WandB artifacts.
   - **Testing Parameters**: Choose the number of episodes for testing. Enable rendering if desired, setting the frequency as needed.
   - **Initiate Testing**: Begin the test by clicking "Start Test".

### Hyperparameter Search
5. **Conducting Hyperparameter Search**: Use Weights & Biases for running hyperparameter searches.
   - **Sweep Method**: Select your preferred sweep method.
   - **WandB Project Selection**: Choose the WandB project for the sweep.
   - **Sweep Parameters**: Input the necessary parameters for conducting the sweep.
   - **Environment and Saving**: Select the environment for the agents and a directory for saving the best model.
   - **Model and Hyperparameter Selection**: Choose the models for the sweep and specify the hyperparameter values for each selected agent.
   - **Start Sweeps**: Begin the hyperparameter sweep by clicking "Start Sweeps".

### WandB Utilities
6. **Weights & Biases Utilities (WIP)**: This section provides utilities for interacting with Weights & Biases.
   - Currently under development, this feature will allow you to log in to WandB and perform various project-level actions, like managing runs, artifacts, and sweeps.


## Contact
For any inquiries or contributions, please contact [Jason Hayes](mailto:jasonhayes1987@gmail.com).
