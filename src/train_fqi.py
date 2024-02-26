from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import pickle
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
        self.batch_size = 32
        self.training_iterations = 8
        self.gamma = 0.99
        self.max_episode = 200

    def act(self, observation, use_random=False):
        return np.random.choice(4) if use_random else self.greedy_action(observation)

    def save(self, path):
        with open("rf_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open("rf_model.pkl", "rb") as f:
            self.model = pickle.load(f)

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
        A = np.array(A).reshape((-1, 1))
        R = np.array(R)
        S2 = np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D

    def train(self):
        Qfunction = None
        episode = 0
        episode_cum_reward = 0
        state, reward, done, trunc, _ = env.reset()
        S, A, R, S2, D = self.collect_samples(env)
        SA = np.append(S, A, axis=1)
        for episode in range(self.max_episode):
            for iter in tqdm(range(self.training_iterations)):
                if iter == 0:
                    value = R.copy()
                else:
                    Q2 = np.zeros((self.batch_size, 4))
                    for a2 in range(4):
                        A2 = a2 * np.ones((S.shape[0], 1))
                        S2A2 = np.append(S2, A2, axis=1)
                        Q2[:, a2] = Qfunction.predict(S2A2)
                    max_Q2 = np.max(Q2, axis=1)
                    value = R + self.gamma * (1 - D) * max_Q2
                Q = RandomForestRegressor()
                Q.fit(SA, value)
                Qfunction = Q
            self.model = Q
            while True:
                action = self.act(state)
                next_state, reward, done, trunc, _ = env.step(action)
                episode_cum_reward += reward
                if done or trunc:
                    print(
                        "Episode ",
                        "{:3d}".format(episode),
                        ", episode return ",
                        "{:4.1f}".format(episode_cum_reward),
                        sep="",
                    )
                    state, _ = env.reset()
                    episode_cum_reward = 0
                    break
                else:
                    state = next_state

    def greedy_action(self, s):
        Qsa = []
        for a in range(4):
            sa = np.append(s, a).reshape(1, -1)
            Qsa.append(self.model.predict(sa))
        return np.argmax(Qsa)


############################################################
############################################################
############################################################


rf = ProjectAgent()
rf.train()
