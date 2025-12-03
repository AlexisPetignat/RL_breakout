import typing as t
import gymnasium as gym
from gymnasium.wrappers import HumanRendering, RecordVideo, RecordEpisodeStatistics
import numpy as np
import ale_py
import os
import glob
import time
from qlearning import QLearningAgent
from model import DQN, preprocess
from upscaler import UpscaleRender
from torch.utils.tensorboard import SummaryWriter

DISPLAY = False
TRAIN = False


env = gym.make(
    "ALE/Breakout-v5", render_mode="rgb_array", repeat_action_probability=0.0
)
env = UpscaleRender(env, scale=3)
writer = SummaryWriter("runs/training")

n_actions = env.action_space.n  # type: ignore
recording_epoch = 250
best_score = -float("inf")
TARGET_UPDATE_INTERVAL = 10000
FRAME_BETWEEN_TRAIN = 3

# Counters
qlearning_rewards = []
total_frames = 0

# env = RecordVideo(
#     env,
#     video_folder="qlearning-agent",  # Folder to save videos
#     name_prefix="eval",  # Prefix for video filenames
#     episode_trigger=lambda x: True,  # Record every episode
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
        if epoch % FRAME_BETWEEN_TRAIN == 0 and TRAIN:
            agent.thinkAboutWhatHappened(
                update=(total_frames + epoch) % TARGET_UPDATE_INTERVAL == 0
            )
        if done:
            return (total_reward, epoch)

        s = next_s
        # END SOLUTION

    return (total_reward, t_max)


# directory where RecordVideo writes videos
VIDEO_FOLDER = "qlearning-agent"
VIDEO_GLOB = os.path.join(VIDEO_FOLDER, "*.mp4")

# best score seen among saved videos
best_score = -float("inf")

M = 90000
for i in range(M):
    if i % 1000 == -1 or DISPLAY:
        env1 = HumanRendering(env)
    else:
        env1 = env

    # --- remember current set of video files before running episode ---
    before_files = set(glob.glob(VIDEO_GLOB))

    # Train / play (TRAIN is False in your config so thinkAboutWhatHappened won't run)
    reward, frames = play_and_train(env1, agent)

    # small pause to allow the recorder to flush the file (usually not necessary,
    # but helps if filesystem delays occur)
    time.sleep(0.1)

    # --- find new files created by the recorder for this episode ---
    after_files = set(glob.glob(VIDEO_GLOB))
    new_files = sorted(list(after_files - before_files), key=os.path.getmtime)

    # Log reward to tensorboard
    writer.add_scalar("Reward/Episode", reward, i)
    writer.add_scalar("Epsilon", agent.epsilon, i)
    writer.add_scalar("Reward/100-episode-mean", np.mean(qlearning_rewards[-100:]), i)

    # Update counters & print
    qlearning_rewards.append(reward)
    total_frames += frames
    print(f"Epoch {i}: mean reward", np.mean(qlearning_rewards[-100:]))
    if i % 10 == 0:
        print(f"Total frames: {total_frames}")
        print(f"Epsilon: {agent.epsilon}")

    # if len(new_files) == 0:
    #     print(f"No new video file detected for episode {i} (reward={reward}).")
    # else:
    #     # Decide whether to keep or remove
    #     if reward >= best_score:
    #         # New best -> keep files and update best_score
    #         best_score = reward
    #         print(
    #             f" Episode {i} saved as new best: reward={reward}. Kept {len(new_files)} file(s)."
    #         )
    #         for fpath in new_files:
    #             base = os.path.basename(fpath)
    #             dirname = os.path.dirname(fpath)
    #             new_name = f"best_{int(best_score):04d}_ep{i}_{base}"
    #             new_path = os.path.join(dirname, new_name)
    #             try:
    #                 os.rename(fpath, new_path)
    #             except Exception:
    #                 # If rename fails (rare), keep as-is
    #                 pass
    #     else:
    #         for fpath in new_files:
    #             try:
    #                 os.remove(fpath)
    #             except Exception as e:
    #                 print(f"Warning: failed to remove {fpath}: {e}")
    #         print(
    #             f"Episode {i} (reward={reward}) not better than best ({best_score}). Deleted {len(new_files)} file(s)."
    #         )
