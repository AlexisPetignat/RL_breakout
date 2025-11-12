from collections import defaultdict
import random
import typing as t
import numpy as np
import gymnasium as gym
import torch
from model import DQN, preprocess


Action = int
State = np.ndarray
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[int, t.DefaultDict[Action, float]]
N = 10000


class QLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        legal_actions: t.List[Action],
        model: DQN,
        batch_size: int = 32,
    ):
        """
        Q-Learning Agent

        You shoud not use directly self._qvalues, but instead of its getter/setter.
        """
        self.legal_actions = legal_actions
        self._qvalues: QValues = defaultdict(lambda: defaultdict(int))
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(torch.device(self.device))
        self.target_model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.D: list[tuple[State, Action, float, State, bool]] = []
        self.frame_buffer = np.zeros((4, 84, 84), dtype=np.float32)
        self.batch_size = batch_size

        self.target_model.load_state_dict(self.model.state_dict())

    def updateBuffer(self, new_state: State):
        # Update the frame buffer
        buffer_state = preprocess(new_state)
        self.frame_buffer[:-1] = self.frame_buffer[1:]
        self.frame_buffer[-1] = buffer_state

    def get_qvalue(self, state_buffer: State, action: Action) -> float:
        """
        Returns the Q value for (state, action)
        """
        # Run prediction
        pred_buffer = torch.tensor(state_buffer, device=self.device).unsqueeze(0)
        res = self.model.forward(pred_buffer)
        return res[0][action].item()

    def set_qvalue(
        self,
        state: State,
        action: Action,
        value: float,
        next_state: State,
        next_state_terminal: bool,
    ):
        """
        Sets the Qvalue for [state,action] to the given value
        """

        # Store a 4 frame buffer for the states
        buffer_next = self.frame_buffer.copy()

        buffer_next[:-1] = buffer_next[1:]
        buffer_next[-1] = preprocess(next_state)

        # Add to memory
        experience = (
            self.frame_buffer,
            action,
            value,
            buffer_next,
            next_state_terminal,
        )
        if len(self.D) == N:
            self.D.pop(0)

        if value != 0:
            for i in range(30):
                self.D.append(experience)
        self.D.append(experience)

    def get_value(self, state: State) -> float:
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_a Q(s, a) over possible actions.
        """
        value = 0.0
        # BEGIN SOLUTION
        value = float("-inf")
        for action in self.legal_actions:
            value = max(value, self.get_qvalue(state, action))
        # END SOLUTION
        return value

    def update(
        self,
        state: State,
        action: Action,
        reward: t.SupportsFloat,
        next_state: State,
        next_state_terminal: bool,
    ):
        """
        You should do your Q-Value update here:

           TD_target(s, a, r, s') = r + gamma * V(s')
           TD_error(s, a, r, s') = TD_target(s, a, r, s') - Q_old(s, a)
           Q_new(s, a) := Q_old(s, a) + learning_rate * TD_error(s, a, R(s, a), s')
        """
        # Update buffer
        self.updateBuffer(next_state)

        # Update the q-value
        # if reward != 0:
        # print(f"Reward: {reward}, action: {action}")
        self.set_qvalue(state, action, reward, next_state, next_state_terminal)

    def get_best_action(self) -> Action:
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_q_values = [
            self.get_qvalue(self.frame_buffer, action) for action in self.legal_actions
        ]
        index = np.argmax(possible_q_values)
        best_action = self.legal_actions[index]
        return best_action

    def get_action(self) -> Action:
        """
        Compute the action to take in the current state, including exploration.

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        action = self.legal_actions[0]

        # Pick exploration with probablity epsilon
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.legal_actions)

        # Else pick exploitation by returning the best action
        else:
            action = self.get_best_action()

        return action

    def thinkAboutWhatHappened(self):
        if len(self.D) < self.batch_size:
            return  # not enough data yet

        # Update model
        self.target_model.load_state_dict(self.model.state_dict())

        # Learning parameters
        batch_size = self.batch_size
        batch = random.sample(self.D, batch_size)
        loss_fn = torch.nn.MSELoss()

        # Prepare device (GPU if available)
        device = self.device

        # Unpack batch and convert to tensors
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=device
        )
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        # Predict Q-values for current and next states
        q_pred = self.model(states)
        q_next = self.target_model(next_states).detach()

        # Select Q-values for chosen actions
        q_pred_action = q_pred.gather(1, actions)

        # Compute target: y_j = r + gamma * max_a' Q(s', a') (only if not done)
        max_next_q = q_next.max(1)[0].unsqueeze(1)
        y_j = rewards + self.gamma * (1 - dones) * max_next_q

        # Compute loss and optimize
        loss = loss_fn(q_pred_action, y_j)
        # print("loss:", loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
