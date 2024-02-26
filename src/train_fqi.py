from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


############################################################
############################################################
############################################################


class ProjectAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 1064
        self.training_iterations = 128
        self.gamma = 0.98

    def act(self, observation, use_random=False):
        if use_random or np.random.rand() <= self.epsilon_min:
            return np.random.choice(4)
        return self.greedy_action(self.model, observation)

    def save(self, path):
        torch.save(self.model.state_dict(), self.path)

    def load(self):
        self.model.load_state_dict(torch.load(self.path))

    def collect_samples(self, env):
        s, _ = env.reset()
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(self.batch_size)):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D

    def rf_fqi(self, S, A, R, S2, D):
        Qfunctions = []
        SA = np.append(S, A, axis=1)
        for iter in tqdm(range(self.training_iterations)):
            if iter == 0:
                value = R.copy()
            else:
                Q2 = np.zeros((self.batch_size, 4))
                for a2 in range(4):
                    A2 = a2 * np.ones((S.shape[0], 1))
                    S2A2 = np.append(S2, A2, axis=1)
                    Q2[:, a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = R + self.gamma * (1 - D) * max_Q2
            Q = RandomForestRegressor()
            Q.fit(SA, value)
            Qfunctions.append(Q)
        return Qfunctions

    def greedy_action(self, model, state):
        with torch.no_grad():
            Q = model(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()


############################################################
############################################################
############################################################


agent = ProjectAgent()
agent.train(env, 5)


###########################################################
###########################################################
###########################################################

