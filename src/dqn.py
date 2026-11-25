'''
DQN implementation
'''

# Import required libraries
from copy import deepcopy
import numpy as np
import torch

# Import custom modules
from scripts import lab09_starter as S


# Define the DQN target function
@torch.no_grad()
def dqn_target(rewards, next_q_values, terminated, gamma):
    'Compute Bellman targets for a batch of transitions.'
    # Keep the best next-state value unless the transition truly terminated.
    best_next_q = next_q_values.max(dim = 1).values
    bootstrap_values = best_next_q.masked_fill(terminated, 0.0)

    return rewards + gamma * bootstrap_values


# Define the Huber loss function
def huber_loss(predictions, targets):
    'Return the per-sample Huber loss with a threshold of 1.'
    absolute_errors = (predictions - targets).abs()

    # Apply a quadratic penalty up to 1 and a linear penalty above 1
    return torch.where(absolute_errors <= 1.0, 0.5 * absolute_errors.square(), absolute_errors - 0.5)


# Define the Replay Buffer class
class ReplayBuffer:
    'Store transitions in a fixed-capacity circular buffer.'

    def __init__(self, capacity, obs_dim):
        # Preallocate storage for each transition field
        self.states = np.empty((capacity, obs_dim), dtype = np.float32)
        self.actions = np.empty(capacity, dtype = np.int64)
        self.rewards = np.empty(capacity, dtype = np.float32)
        self.next_states = np.empty((capacity, obs_dim), dtype = np.float32)
        self.terminated = np.empty(capacity, dtype = np.bool_)
        self.capacity = capacity
        self.size = 0
        self.position = 0

    def add(self, state, action, reward, next_state, terminated):
        # Overwrite the oldest transition when the buffer is full
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminated[index] = terminated

        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        # Draw a uniform batch without replacement from the populated slots
        indices = rng.choice(self.size, size = batch_size, replace = False)
        return self._batch(indices)

    def chronological_batch(self):
        # Read the full buffer from the oldest transition to the newest
        indices = (self.position + np.arange(self.capacity)) % self.capacity
        return self._batch(indices)

    def _batch(self, indices):
        # Convert the selected transition fields to PyTorch tensors
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.terminated[indices]),
        )

# Define the epsilon at step function
def epsilon_at_step(step):
    'Return the exploration rate for a one-indexed environment step.'
    # Decay linearly from 1.0 at step 1 to 0.05 at step 8,000.
    fraction = min((step - 1) / 7_999, 1.0)
    return (1.0 - fraction) + 0.05 * fraction

# Define the train DQN function
def train_dqn(seed, steps = S.BUDGET, use_target_network = True, use_replay = True, replay_capacity = 20_000):
    'Train one DQN configuration and return episode logs and Q measurements.'
    # Seed the environment and keep exploration and replay sampling independent
    S.set_seed(seed)
    env = S.CartPole(seed = seed)
    action_rng = np.random.default_rng(seed)
    replay_rng = np.random.default_rng(seed + 10_000)

    # Use a separate target copy for the baseline and the online network for the ablation
    online_network = S.QNet()
    target_network = deepcopy(online_network) if use_target_network else online_network
    optimizer = torch.optim.Adam(online_network.parameters(), lr = 0.001)

    # The no-replay configuration holds one chronological batch at a time
    batch_size = 64
    capacity = replay_capacity if use_replay else batch_size
    replay = ReplayBuffer(capacity = capacity, obs_dim = env.obs_dim)
    probes = S.probe_states()

    episodes = []
    q_measurements = []
    state = env.reset()
    episode_return = 0.0

    for step in range(1, steps + 1):
        # Explore with probability epsilon and otherwise take the greedy action
        if action_rng.random() < epsilon_at_step(step):
            action = int(action_rng.integers(env.n_actions))
        else:
            with torch.no_grad():
                action = online_network(torch.from_numpy(state)).argmax().item()

        # Store the actual next state and preserve true termination for bootstrapping
        next_state, reward, terminated, truncated = env.step(action)
        replay.add(state, action, reward, next_state, terminated)
        episode_return += reward
        state = next_state

        # Log the finished episode before resetting the environment
        if terminated or truncated:
            episodes.append((step, episode_return))
            state = env.reset()
            episode_return = 0.0

        # Begin replay updates after 500 steps or update once per full chronological batch
        ready_to_update = step > 500 if use_replay else step % batch_size == 0

        if ready_to_update:
            if use_replay:
                batch = replay.sample(batch_size, replay_rng)
            else:
                batch = replay.chronological_batch()

            states, actions, rewards, next_states, terminals = batch

            # Gather the predicted Q-value for each action taken
            predictions = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Build detached targets and update the online network
            with torch.no_grad():
                next_q_values = target_network(next_states)
                targets = dqn_target(rewards, next_q_values, terminals, gamma = 0.99)

            loss = huber_loss(predictions, targets).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_norm = 10.0)
            optimizer.step()

        # Refresh the separate target network every 250 environment steps
        if use_target_network and step % 250 == 0:
            target_network.load_state_dict(online_network.state_dict())

        # Measure mean greedy Q and maximum absolute Q on the fixed probe states
        if step % 1_000 == 0:
            with torch.no_grad():
                probe_q = online_network(probes)
                mean_max_q = probe_q.max(dim = 1).values.mean().item()
                max_abs_q = probe_q.abs().max().item()

            q_measurements.append((step, mean_max_q, max_abs_q))

    return episodes, q_measurements