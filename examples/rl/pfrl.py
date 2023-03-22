# %%
import pfrl
import torch
import torch.nn
import gym
import numpy as np
# %%
env = gym.make("CartPole-v1")
print("observation space:", env.observation_space)
print("action space:", env.action_space)

obs = env.reset()
print("initial obs:", obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print("next obs:", obs)
print("reward:", r)
print("done:", done)
print("info:", info)

# env.render()

# %%
class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 50)
        self.l2 = torch.nn.Linear(50, 50)
        self.l3 = torch.nn.Linear(50, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)

obs_size = env.observation_space.low.size
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)

# %%
q_func2 = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, n_actions),
    pfrl.q_functions.DiscreteActionValueHead(),
)

# %%
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
# %%
gamma = 0.9
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample
)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10**6)
phi = lambda x: x.astype(np.float32, copy=False)

gpu = 0
agent = pfrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer, replay_start_size=500,
    update_interval=1, target_update_interval=100, phi=phi, gpu=gpu
)

# %%
n_episodes = 300
max_episode_len = 200
for i in range(1, n_episodes+1):
    obs = env.reset()
    R = 0
    t = 0
    while True:
        env.render()
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = (t == max_episode_len)
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
print('Finished.')

# %%
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

pfrl.experiments.train_agent_with_evaluation(
    agent, env, steps=2000, 
    eval_n_steps=None,
    eval_n_episodes=10,
    train_max_episode_len=200, eval_interval=1000, outdir='result'
)
# %%
