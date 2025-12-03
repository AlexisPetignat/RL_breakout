from collections import defaultdict
import random
import typing as t
import numpy as np
import gymnasium as gym
import torch
from model import DQN, preprocess
import copy


Action = int
State = np.ndarray
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[int, t.DefaultDict[Action, float]]

N = int(5 * 1e4)  # Memory Size
PATH = "./parameters.pt2"  # Path for model parameters


class QLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        decay: int,
        epsilon_min: float,
        gamma: float,
        legal_actions: t.List[Action],
        model: DQN,
        batch_size: int = 32,
        retrain: bool = False,
    ):
        """
        Q-Learning Agent

        You shoud not use directly self._qvalues, but instead of its getter/setter.
        """

        if not retrain:
            model.load_state_dict(torch.load(PATH, weights_only=True))

        self.legal_actions = legal_actions
        self._qvalues: QValues = defaultdict(lambda: defaultdict(int))
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay_steps = decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(torch.device(self.device))
        self.target_model = copy.deepcopy(self.model).to(torch.device(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.D: list[tuple[State, Action, float, State, bool]] = []
        self.frame_buffer = np.zeros((4, 84, 84), dtype=np.float32)
        self.batch_size = batch_size

    def updateEpsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (1.0 - self.epsilon_min) / self.epsilon_decay_steps

    def initializeBuffer(self, initial_state: State):
        state_processed = preprocess(initial_state)
        state_processed_buffered = np.expand_dims(state_processed, 0)
        self.frame_buffer = np.repeat(state_processed_buffered, 4, axis=0)

    def updateBuffer(self, new_state: State):
        # Update the frame buffer
        buffer_state = preprocess(new_state)
        self.frame_buffer[:-1] = self.frame_buffer[1:]
        self.frame_buffer[-1] = buffer_state

    def get_qvalues(self, state_buffer: State):
        """
        Returns the Q value for (state, action)
        """
        # Run prediction
        pred_buffer = torch.tensor(
            state_buffer, device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        # Predict without gradient
        self.model.eval()
        with torch.no_grad():
            res = self.model(pred_buffer)
        self.model.train()

        return res[0]

    def get_qvalue(self, state_buffer: State, action: Action) -> float:
        """
        Returns the Q value for (state, action)
        """
        qvals = self.get_qvalues(state_buffer)  # single forward
        return qvals[action].item()

    def get_value(self, state: State) -> float:
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_a Q(s, a) over possible actions.
        """
        qvals = self.get_qvalues(state)
        return float(qvals.max().item())

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

        # # Define valiables for the update formula
        # q_old = self.get_qvalue(self.frame_buffer, action)
        # gamma = self.gamma

        # # Compute fake next step buffer
        # buffer_next = self.frame_buffer.copy()
        # buffer_next[:-1] = buffer_next[1:]
        # buffer_next[-1] = preprocess(next_state)
        # next_step_value = self.get_value(buffer_next)

        # # Compute target error
        # TD_target = float(reward)
        # if not next_state_terminal:
        #     TD_target += gamma * next_step_value
        # TD_error = abs(TD_target - q_old)
        # # q_value = q_old + learning_rate * TD_error

        # # If high error, add it several times so it is seen often during training
        # if TD_error > 1:
        #     self.set_qvalue(
        #         state, action, reward, next_state, next_state_terminal, repeat=30
        #     )
        # elif TD_error > 0.1:
        #     self.set_qvalue(
        #         state, action, reward, next_state, next_state_terminal, repeat=5
        #     )
        # else:
        #     self.set_qvalue(state, action, reward, next_state, next_state_terminal)

        # Current state buffer is already in self.frame_buffer
        current_buffer = self.frame_buffer.copy()

        # Update buffer to next state FIRST
        self.updateBuffer(next_state)

        # Now frame_buffer contains the next state
        next_buffer = self.frame_buffer.copy()

        # Store the experience with correct buffers
        experience = (
            current_buffer,
            action,
            float(reward),
            next_buffer,
            next_state_terminal,
        )
        self.D.append(experience)

        # Remove old experiences if memory is full
        if len(self.D) > N:
            self.D.pop(0)

    def get_best_action(self) -> Action:
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_q_values = self.get_qvalues(self.frame_buffer)
        possible_q_values = [a.item() for a in possible_q_values]
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

    def thinkAboutWhatHappened(self, update: bool):
        if len(self.D) < self.batch_size:
            return  # not enough data yet

        # Update model
        if update:
            self.target_model.load_state_dict(self.model.state_dict())
            torch.save(self.model.state_dict(), PATH)

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

        # Select Q-values for chosen actions
        q_pred_action = q_pred.gather(1, actions)

        # q_next = self.target_model(next_states).detach()
        # max_next_q = q_next.max(1)[0].unsqueeze(1)
        # y_j = rewards + self.gamma * (1 - dones) * max_next_q

        # --- Double DQN target computation ---

        # 1) Online network chooses the best actions
        q_next_online = self.model(next_states).detach()  # Q_online(s')
        best_actions = q_next_online.argmax(dim=1, keepdim=True)

        # 2) Target network evaluates those actions
        q_next_target = self.target_model(next_states).detach()
        selected_target_q = q_next_target.gather(1, best_actions)

        # 3) Build Double DQN target
        y_j = rewards + self.gamma * (1 - dones) * selected_target_q

        # Compute loss and optimize
        loss = loss_fn(y_j, q_pred_action)
        # print("Loss:", loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
