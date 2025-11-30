import typing as t
import gymnasium as gym
from gymnasium.wrappers import HumanRendering, RecordVideo, RecordEpisodeStatistics
import numpy as np
import ale_py
from qlearning import QLearningAgent
from model import DQN, preprocess
from upscaler import UpscaleRender

DISPLAY = False


env = gym.make(
    "ALE/Breakout-v5", render_mode="rgb_array", repeat_action_probability=0.0
)
env = UpscaleRender(env, scale=3)

n_actions = env.action_space.n  # type: ignore
recording_epoch = 250
TARGET_UPDATE_INTERVAL = 10000
FRAME_BETWEEN_TRAIN = 3

# Counters
qlearning_rewards = []
total_frames = 0

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
    learning_rate=0.00025,
    epsilon=0.05,
    decay=30000,
    epsilon_min=0.05,
    gamma=0.99,
    legal_actions=list(range(n_actions)),
    model=model,
    batch_size=128,
    retrain=False,
)


def play_and_train(
    env: gym.Env, agent: QLearningAgent, t_max=int(1e4)
) -> tuple[float, int]:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0

    # State at start of the game
    s, _ = env.reset()
    agent.initializeBuffer(s)
    agent.updateEpsilon()

    for epoch in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action()

        next_s, r, done, _, _ = env.step(a)

        # Clip reward just in case
        r = np.clip(r, -1, 1)

        # Train agent for state s
        # BEGIN SOLUTION
        total_reward += r

        agent.update(s, a, r, next_s, done)

        # Train only sometimes
        if epoch % FRAME_BETWEEN_TRAIN == 0:
            agent.thinkAboutWhatHappened(
                update=(total_frames + epoch) % TARGET_UPDATE_INTERVAL == 0
            )
        if done:
            return (total_reward, epoch)

        s = next_s
        # END SOLUTION

    return (total_reward, t_max)


M = 90000
for i in range(M):
    if i % 1000 == -1  or DISPLAY:
        env1 = HumanRendering(env)
    else:
        env1 = env

    # Train
    reward, frames = play_and_train(env1, agent)

    # Update counters
    qlearning_rewards.append(reward)
    total_frames += frames
    print(f"Epoch {i}: mean reward", np.mean(qlearning_rewards[-100:]))
    if i % 10 == 0:
        print(f"Total frames: {total_frames}")
        print(f"Epsilon: {agent.epsilon}")

assert np.mean(qlearning_rewards[-100:]) > 0.0
