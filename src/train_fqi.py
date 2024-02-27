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
        self.collect_size = 600
        self.Q_iterations = 30
        self.nb_epochs = 50
        self.gamma = 0.98
        ### END PARAMS
        self.S = []
        self.A = []
        self.R = []
        self.S2 = []
        self.D = []
        self.Q = None

    def collect_samples(self, env, randomness=0.0):
        s, _ = env.reset()
        for _ in tqdm(range(self.collect_size)):
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
        S = np.array(self.S)
        A = np.array(self.A).reshape((-1, 1))
        R = np.array(self.R)
        S2 = np.array(self.S2)
        D = np.array(self.D)
        SA = np.append(S, A, axis=1)
        Q = self.Q
        for iter in tqdm(range(self.Q_iterations)):
            if iter == 0 and Qfunction == None:
                value = R.copy()
            else:
                Q2 = np.zeros((self.batch_size, 4))
                for a2 in range(4):
                    A2 = a2 * np.ones((S.shape[0], 1))
                    S2A2 = np.append(S2, A2, axis=1)
                    Q2[:, a2] = Qfunction.predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = R + self.gamma * (1 - D) * max_Q2
            Q = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
            Q.fit(SA, value)
            Qfunction = Q
        self.model = Q

    def greedy_action(self, s):
        Qsa = []
        for a in range(4):
            sa = np.append(s, a).reshape(1, -1)
            Qsa.append(self.model.predict(sa))
        return np.argmax(Qsa)
    
    def train(self):
        self.collect_samples(env, randomness=0.0)
        self.rf_fqi()
        print(0, evaluate_HIV(agent=self, nb_episode=5) / 10e6)
        self.save("rf_model.pkl")
        for epoch in range(self.nb_epochs):
            self.collect_samples(env, randomness=0.1)
            self.rf_fqi()
            print(epoch + 1, evaluate_HIV(agent=self, nb_episode=5) / 10e6)
            self.save("rf_model.pkl")

    def act(self, observation, use_random=False):
        return np.random.choice(4) if use_random else self.greedy_action(observation)

    def save(self, path):
        with open("rf_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open("rf_model.pkl", "rb") as f:
            self.model = pickle.load(f)

############################################################
############################################################
############################################################


rf = ProjectAgent()
rf.train()
