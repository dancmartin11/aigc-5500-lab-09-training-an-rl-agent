'''
Policy Gradient implementation
'''

# Import required libraries
import torch

# Define the PPO clipped objective function
def ppo_clipped_objective(ratios, advantages, clip = 0.2):
    'Return the clipped PPO objective for each sample.'

    # Limit the policy ratios to the interval around 1
    clipped_ratios = ratios.clamp(1.0 - clip, 1.0 + clip)

    # Select the smaller objective for each sample
    return torch.minimum(ratios * advantages, clipped_ratios * advantages)


# Define the returns to go function
def returns_to_go(rewards, gamma):
    'Return the discounted future rewards from each step.'
    
    discounted_returns = [0.0] * len(rewards)
    running_return = 0.0

    # Work backward so each step can reuse the return from the next step
    for step in range(len(rewards) - 1, -1, -1):
        running_return = rewards[step] + gamma * running_return
        discounted_returns[step] = running_return

    return discounted_returns