"""Adds a custom callback to the keras model to update streamlit elements"""

# imports
from tensorflow.keras.callbacks import Callback


class StreamlitCallback(Callback):
    """Custom callback to update streamlit elements"""

    def __init__(self, progress_bar, terminal_output, num_episodes):
        """Initializes the callback.

        Args:
            progress_bar (streamlit.progress): Streamlit progress bar element.
            terminal_output (streamlit.terminal): Streamlit terminal output element.
            num_episodes (int): Number of episodes to train for.

        """

        super().__init__()
        self.progress_bar = progress_bar
        self.terminal_output = terminal_output
        self.num_episodes = num_episodes
        self.epoch = 0

    def on_batch_end(self, batch, logs=None):
        """Updates the progress bar and terminal output."""

    def On_epoch_begin(self, epoch, logs=None):
        """Updates the progress bar and terminal output."""

    def on_epoch_end(self, epoch, logs=None):
        """Updates the progress bar and terminal output."""
        # udpate epoch number
        self.epoch += 1
        # Update Streamlit elements
        self.progress_bar.progress(self.epoch / self.num_episodes)
        message = f"### Episode {self.epoch}\nEpisode Reward: {logs['episode_reward']}\nAVG Reward: {logs['avg_reward']}"
        self.terminal_output.markdown(message)

    def on_train_begin(self, logs=None):
        """Updates the progress bar and terminal output."""
        self.terminal_output.markdown("### Training in Progress")
        self.epoch = 0

    def on_train_end(self, logs=None):
        """Updates the progress bar and terminal output."""
        self.terminal_output.markdown(
            f"### Training Complete\nFinal Reward: {logs['episode_reward']}"
        )

    def on_test_begin(self, logs=None):
        """Updates the progress bar and terminal output."""
        self.terminal_output.markdown("### Testing in Progress")

    def on_test_end(self, logs=None):
        """Updates the progress bar and terminal output."""
        self.terminal_output.markdown(
            f"### Testing Complete\nFinal Reward: {logs['episode_reward']}"
        )

    def on_test_batch_end(self, batch, logs=None):
        """Updates the progress bar and terminal output."""
        # udpate epoch number
        self.epoch += 1
        # Update Streamlit elements
        self.progress_bar.progress(self.epoch / self.num_episodes)
        self.terminal_output.markdown(
            f"### Episode {self.epoch}\nEpisode Reward: {logs['episode_reward']}"
        )
