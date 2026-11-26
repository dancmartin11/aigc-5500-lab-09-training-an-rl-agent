'''
Policy Gradient implementation
'''

# Import required libraries
import numpy as np
import torch

# Import custom modules
from scripts import lab09_starter as S

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

# Define the train REINFORCE function
def train_reinforce(seed, steps = S.BUDGET):
    'Train REINFORCE with a learned value baseline and return the episode log.'
    S.set_seed(seed)
    env = S.CartPole(seed = seed)

    # Build separate networks and optimizers for the policy and baseline
    policy = S.mlp(env.obs_dim, env.n_actions)
    value_network = S.mlp(env.obs_dim, 1)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr = 0.003)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr = 0.01)

    episodes = []
    episode_states, episode_actions, episode_rewards = [], [], []
    batch_states, batch_actions, batch_returns = [], [], []
    episodes_in_batch = 0
    state = env.reset()

    for step in range(1, steps + 1):
        # Sample an action from the current policy
        with torch.no_grad():
            logits = policy(torch.from_numpy(state))
            action = torch.distributions.Categorical(logits = logits).sample().item()

        next_state, reward, terminated, truncated = env.step(action)
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        state = next_state

        if not (terminated or truncated):
            continue

        # Log the completed episode and compute its observed returns to go
        episodes.append((step, sum(episode_rewards)))
        batch_states.extend(episode_states)
        batch_actions.extend(episode_actions)
        batch_returns.extend(returns_to_go(episode_rewards, gamma = 0.99))
        episodes_in_batch += 1

        episode_states, episode_actions, episode_rewards = [], [], []
        state = env.reset()

        if episodes_in_batch < 8:
            continue

        # Compute advantages with the baseline before either network is updated
        states = torch.from_numpy(np.stack(batch_states))
        actions = torch.tensor(batch_actions, dtype = torch.long)
        returns = torch.tensor(batch_returns, dtype = torch.float32)
        values = value_network(states).squeeze(1)
        advantages = returns - values.detach()
        advantages = advantages / advantages.std(unbiased = False).clamp_min(1e-8)

        # Weight the selected actions' log probabilities by their advantages
        distribution = torch.distributions.Categorical(logits = policy(states))
        policy_loss = -(distribution.log_prob(actions) * advantages).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Fit the baseline to the observed returns
        value_loss = (values - returns).square().mean()
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        batch_states, batch_actions, batch_returns = [], [], []
        episodes_in_batch = 0

    return episodes


# Define the generalized advantages function
@torch.no_grad()
def generalized_advantages(rewards, values, next_values, terminated, episode_ends, gamma = 0.99, lam = 0.95):
    'Compute generalized advantage estimates for one PPO rollout.'
    # Keep next-state value at time-limit truncations and zero it at true terminals
    bootstrap_values = next_values.masked_fill(terminated, 0.0)
    deltas = rewards + gamma * bootstrap_values - values

    # Stop the backward trace at either kind of episode boundary
    continuation = (~episode_ends).to(values.dtype)
    advantages = torch.empty_like(values)
    running_advantage = 0.0

    for index in range(len(rewards) - 1, -1, -1):
        running_advantage = (deltas[index] + gamma * lam * continuation[index] * running_advantage)
        advantages[index] = running_advantage

    return advantages

# Define the train PPO function
def train_ppo(seed, steps = S.BUDGET):
    'Train PPO with GAE and return the episode log.'
    S.set_seed(seed)
    env = S.CartPole(seed = seed)

    # Use the starter's networks and one optimizer for policy and value updates
    policy = S.mlp(env.obs_dim, env.n_actions)
    value_network = S.mlp(env.obs_dim, 1)
    parameters = list(policy.parameters()) + list(value_network.parameters())
    optimizer = torch.optim.Adam(parameters, lr = 0.0003)

    episodes = []
    state = env.reset()
    episode_return = 0.0
    environment_steps = 0

    while environment_steps < steps:
        # Shorten the last rollout to stay within the exact environment-step budget
        rollout_length = min(1_024, steps - environment_steps)
        rollout_states, rollout_next_states = [], []
        rollout_actions, rollout_rewards, old_log_probs = [], [], []
        terminal_flags, episode_end_flags = [], []

        for _ in range(rollout_length):
            # Save the action log probability under the policy that collected it
            with torch.no_grad():
                logits = policy(torch.from_numpy(state))
                distribution = torch.distributions.Categorical(logits = logits)
                sampled_action = distribution.sample()
                old_log_prob = distribution.log_prob(sampled_action).item()

            action = sampled_action.item()
            next_state, reward, terminated, truncated = env.step(action)
            episode_ended = terminated or truncated
            environment_steps += 1
            episode_return += reward

            # Keep the actual next state before a reset replaces it
            rollout_states.append(state)
            rollout_next_states.append(next_state)
            rollout_actions.append(action)
            rollout_rewards.append(reward)
            old_log_probs.append(old_log_prob)
            terminal_flags.append(terminated)
            episode_end_flags.append(episode_ended)

            state = next_state
            if episode_ended:
                episodes.append((environment_steps, episode_return))
                state = env.reset()
                episode_return = 0.0

        # Convert the rollout to tensors once for all six PPO epochs
        states = torch.from_numpy(np.stack(rollout_states))
        next_states = torch.from_numpy(np.stack(rollout_next_states))
        actions = torch.tensor(rollout_actions, dtype = torch.long)
        rewards = torch.tensor(rollout_rewards, dtype = torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype = torch.float32)
        terminated = torch.tensor(terminal_flags, dtype = torch.bool)
        episode_ends = torch.tensor(episode_end_flags, dtype = torch.bool)

        # Freeze the advantages and value targets before optimizer updates begin
        with torch.no_grad():
            values = value_network(states).squeeze(1)
            next_values = value_network(next_states).squeeze(1)
            advantages = generalized_advantages(rewards, values, next_values, terminated, episode_ends)
            value_targets = values + advantages
            advantages = ((advantages - advantages.mean()) / advantages.std(unbiased = False).clamp_min(1e-8))

        for _ in range(6):
            # Shuffle the rollout for each epoch and keep the final partial minibatch
            shuffled_indices = torch.randperm(rollout_length)

            for indices in shuffled_indices.split(128):
                logits = policy(states[indices])
                distribution = torch.distributions.Categorical(logits = logits)
                new_log_probs = distribution.log_prob(actions[indices])

                # Compare the updated policy with the fixed collection policy
                ratios = (new_log_probs - old_log_probs[indices]).exp()
                policy_loss = -ppo_clipped_objective(ratios, advantages[indices], clip = 0.2).mean()

                # Fit the value network to the fixed GAE-based targets
                predicted_values = value_network(states[indices]).squeeze(1)
                value_loss = (predicted_values - value_targets[indices]).square().mean()
                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm = 0.5)
                optimizer.step()

    return episodes