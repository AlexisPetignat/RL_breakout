import typing as t
import gymnasium as gym
from gymnasium.wrappers import HumanRendering, RecordVideo, RecordEpisodeStatistics
import numpy as np
import ale_py
from qlearning import QLearningAgent
from model import DQN, preprocess


env = gym.make("ALE/Pong-v5", render_mode="rgb_array", repeat_action_probability=0.0)

n_actions = env.action_space.n  # type: ignore
recording_epoch = 250

# env = RecordVideo(
#    env,
#    video_folder="qlearning-agent",  # Folder to save videos
#    name_prefix="eval",  # Prefix for video filenames
#    episode_trigger=lambda x: x % recording_epoch == 0,  # Record every episode
# )
# env = RecordEpisodeStatistics(env, buffer_length=1000)


# Model definition for learning process
model = DQN(num_actions=n_actions)

# You can edit these hyperparameters!
agent = QLearningAgent(
    learning_rate=1.0,
    epsilon=0.25,
    gamma=0.99,
    legal_actions=list(range(n_actions)),
    model=model,
    batch_size=32,
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    # State at start of the game

    for epoch in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action()

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        total_reward += r

        agent.update(s, a, r, next_s, done)

        # Train only sometimes
        if epoch % 10 == 0:
            agent.thinkAboutWhatHappened()
        if done:
            break

        s = next_s
        # END SOLUTION

    return total_reward


qlearning_rewards = []

M = 10000
for i in range(M):
    if i % 10 == 0:
        env1 = env  # HumanRendering(env)
        print(
            f"Sum of non zero transitions: {sum(1 for (_, _, r, _, _) in agent.D if r != 0)}"
        )
    else:
        env1 = env
    qlearning_rewards.append(play_and_train(env1, agent, t_max=1000))
    print(f"Epoch {i}: mean reward", np.mean(qlearning_rewards[-100:]))

assert np.mean(qlearning_rewards[-100:]) > 0.0
