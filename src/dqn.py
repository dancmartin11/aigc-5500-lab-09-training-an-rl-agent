''''
DQN implementation
'''

# Import required libraries
import torch
import numpy as np
from copy import deepcopy
from scripts import lab09_starter as S

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

# Define the Replay Buffer class
class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        # Preallocate storage for the five fields in each transition.
        self.states = np.empty((capacity, obs_dim), dtype = np.float32)
        self.actions = np.empty(capacity, dtype = np.int64)
        self.rewards = np.empty(capacity, dtype = np.float32)
        self.next_states = np.empty((capacity, obs_dim), dtype = np.float32)
        self.terminated = np.empty(capacity, dtype = np.bool_)
        self.capacity = capacity
        self.size = 0
        self.position = 0

    def add(self, state, action, reward, next_state, terminated):
        # Overwrite the oldest transition when the buffer is full.
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminated[index] = terminated

        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        # Draw a uniform batch from the populated slots.
        indices = rng.choice(self.size, size = batch_size, replace = False)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.terminated[indices]),
        )


def epsilon_at_step(step):
    # Start at 1.0 on step 1 and reach 0.05 on step 8,000.
    fraction = min((step - 1) / 7_999, 1.0)
    return (1.0 - fraction) + 0.05 * fraction


def train_dqn(seed, steps = S.BUDGET):
    # Seed the environment, network initialization, and sampling.
    S.set_seed(seed)
    env = S.CartPole(seed = seed)
    action_rng = np.random.default_rng(seed)
    replay_rng = np.random.default_rng(seed + 10_000)

    # Initialize the online network and its target copy.
    online_network = S.QNet()
    target_network = deepcopy(online_network)
    optimizer = torch.optim.Adam(online_network.parameters(), lr = 0.001)
    replay = ReplayBuffer(capacity = 20_000, obs_dim = env.obs_dim)
    probes = S.probe_states()

    episodes = []
    q_measurements = []
    state = env.reset()
    episode_return = 0.0

    for step in range(1, steps + 1):
        # Choose a random action with probability epsilon.
        if action_rng.random() < epsilon_at_step(step):
            action = int(action_rng.integers(env.n_actions))
        else:
            with torch.no_grad():
                action = online_network(torch.from_numpy(state)).argmax().item()

        # Store only true termination as the flag that blocks bootstrapping.
        next_state, reward, terminated, truncated = env.step(action)
        replay.add(state, action, reward, next_state, terminated)
        episode_return += reward
        state = next_state

        # Log finished episodes against the environment-step counter.
        if terminated or truncated:
            episodes.append((step, episode_return))
            state = env.reset()
            episode_return = 0.0

        if step > 500:
            # Gather the online Q-value for each action taken in the batch.
            states, actions, rewards, next_states, terminals = replay.sample(64, replay_rng)
            predictions = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Build detached targets and apply one Huber-loss update.
            with torch.no_grad():
                targets = dqn_target(rewards, target_network(next_states), terminals, gamma = 0.99)

            loss = huber_loss(predictions, targets).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_norm = 10.0)
            optimizer.step()

        # Refresh the target copy every 250 environment steps.
        if step % 250 == 0:
            target_network.load_state_dict(online_network.state_dict())

        if step % 1_000 == 0:
            # Record both Q statistics on the same fixed probe states.
            with torch.no_grad():
                probe_q = online_network(probes)
                mean_max_q = probe_q.max(dim = 1).values.mean().item()
                max_abs_q = probe_q.abs().max().item()

            q_measurements.append((step, mean_max_q, max_abs_q))

    return episodes, q_measurements