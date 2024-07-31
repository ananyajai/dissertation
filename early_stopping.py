import torch

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path='checkpoint.pt'):
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): Number of episodes to wait for after the last time the validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the model when improvement is observed.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_value_loss = float('inf')
        self.counter = 0


    def should_stop(self, value_loss, model):
        """
        Checks if early stopping should be triggered and 
        saves the model if there is an improvement.

        Args:
            value_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.best_value_loss - value_loss > self.min_delta:
            # Improvement detected, save the model and reset the counter
            self.checkpoint(value_loss, model)
            self.best_value_loss = value_loss
            self.counter = 0
        else:
            # No improvement detected, increment counter
            self.counter += 1

        return self.counter >= self.patience


    def checkpoint(self, value_loss, model):
        """
        Saves the model state to the specified path.

        Args:
            value_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        torch.save(model.state_dict(), self.path)
        self.best_value_loss = value_loss
