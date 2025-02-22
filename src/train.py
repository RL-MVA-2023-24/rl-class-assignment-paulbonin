from gymnasium.wrappers import TimeLimit
from evaluate import evaluate_HIV, evaluate_HIV_population
from env_hiv import HIVPatient
import numpy as np
import pickle
import random
import os
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor


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


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


class ProjectAgent:
    def __init__(self):
        ### PARAMS
        self.collect_size = 500
        self.Q_iterations = 50
        self.nb_epochs = 40
        self.gamma = 0.99
        ### END PARAMS
        self.S = []
        self.A = []
        self.R = []
        self.S2 = []
        self.D = []
        self.Q = None

    def collect_samples(self, env, nb_samples, randomness=0.0):
        s, _ = env.reset()
        for _ in tqdm(range(nb_samples)):
            if np.random.rand() < randomness:
                a = np.random.choice(4)
            else:
                a = self.greedy_action(s)
            s2, r, done, trunc, _ = env.step(a)
            self.S.append(s)
            self.A.append(a)
            self.R.append(r)
            self.S2.append(s2)
            self.D.append(done)
            if done or trunc:
                s, _ = env.reset()
            else:
                s = s2

    def rf_fqi(self):
        if len(self.S) < 10000:
            S = np.array(self.S)
            A = np.array(self.A).reshape((-1, 1))
            R = np.array(self.R)
            S2 = np.array(self.S2)
            D = np.array(self.D)
            SA = np.append(S, A, axis=1)
        else:
            S = np.array(self.S)[-10000:]
            A = np.array(self.A)[-10000:].reshape((-1, 1))
            R = np.array(self.R)[-10000:]
            S2 = np.array(self.S2)[-10000:]
            D = np.array(self.D)[-10000:]
            SA = np.append(S, A, axis=1)
        for iter in tqdm(range(self.Q_iterations)):
            if iter == 0 and self.Q == None:
                value = R.copy()
            else:
                Q2 = np.zeros((len(S), 4))
                for a2 in range(4):
                    A2 = a2 * np.ones((S.shape[0], 1))
                    S2A2 = np.append(S2, A2, axis=1)
                    Q2[:, a2] = self.Q.predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = R + self.gamma * (1 - D) * max_Q2
            Q = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
            Q.fit(SA, value)
            self.Q = Q

    def greedy_action(self, s):
        Qsa = []
        for a in range(4):
            sa = np.append(s, a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        return np.argmax(Qsa)

    def train(self):
        for epoch in range(self.nb_epochs):
            if epoch == 0:
                self.collect_samples(env, self.collect_size * 10, randomness=1.0)
                self.rf_fqi()
                print(epoch + 1, evaluate_HIV(agent=self, nb_episode=1) / 1e6)
            else:
                self.collect_samples(env, self.collect_size, randomness=0.15)
                self.rf_fqi()
                # seed_everything(seed=42)
                print(epoch + 1, evaluate_HIV(agent=self, nb_episode=1) / 1e6)
            if (epoch + 1) % 5 == 0:
                self.save("XXX.pkl")
                print("Model saved")

    def act(self, observation, use_random=False):
        return np.random.choice(4) if use_random else self.greedy_action(observation)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self):
        with open("model.pkl", "rb") as f:
            self.Q = pickle.load(f)


############################################################
############################################################
############################################################


# rf = ProjectAgent()
# rf.train()
