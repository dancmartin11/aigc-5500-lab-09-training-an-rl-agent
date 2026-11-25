''''
DQN implementation
'''

# Import required libraries
import torch

# Define the DQN target function
@torch.no_grad()
def dqn_target(rewards, next_q_values, terminated, gamma):
    'Compute Bellman targets, bootstrapping unless a transition terminated.'

    # Select the best action value for each next state
    best_next_q = next_q_values.max(dim = 1).values

    # Zero out the bootstrap value only for true terminal transitions
    bootstrap_values = best_next_q.masked_fill(terminated, 0.0)

    # Add the discounted next-state value to the immediate reward
    return rewards + gamma * bootstrap_values


# Define the Huber loss function
def huber_loss(predictions, targets):
    'Return the Huber loss for each prediction, with a threshold of 1.'
    
    absolute_errors = (predictions - targets).abs()

    # Use a quadratic penalty for small errors and a linear penalty above 1
    return torch.where(absolute_errors <= 1.0, 0.5 * absolute_errors.square(), absolute_errors - 0.5)