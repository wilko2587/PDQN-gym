from PDQN import Agent, train_agent
import gym
import gym_platform
import numpy as np

env = gym.make("Platform-v0")

state_size = env.observation_space.spaces[0].shape[0]
action_space = env.action_space
action_size = action_space.spaces[0].n
action_param_sizes = np.array(
    [action_space.spaces[1].spaces[i].shape[0] for i in range(action_size)])
action_param_size = int(action_param_sizes.sum())

actorNet_kwargs = {'hidden_layers': (128,), 'l2': 0, 'lr': 1e-3}
paramNet_kwargs = {'hidden_layers': (128,), 'l2': 0, 'lr': 1e-4}
Nepisodes = 30000

agent = Agent(state_size=state_size,
                  action_size=action_size,
                  action_param_size=action_param_size,
                  actorNet_kwargs=actorNet_kwargs,
                  paramNet_kwargs=paramNet_kwargs,
                  train_start=500,
                  epsilon_decay=0.9995,
                  epsilon_min=0.01,
                  epsilon_bumps=[],
                  memory_size=10000,
                  batch_size=128,
                  gamma=0.9,
                  grad_clipping=10.,
                  stratify_replay_memory=False)

scores = train_agent(env, agent, episodes=Nepisodes, render=False)
# bin the episodes into 500 length bins.
scores_binned = pd.DataFrame(index=np.floor(np.arange(0, len(scores)) / 500.) * 500, columns=['score'], data=scores)
scores_binned = scores_binned.reset_index()
scores_binned = scores_binned.rename(columns={'index': 'episode'})
f = plt.figure()
sns.pointplot(data=scores_binned, y='score', x='episode', errwidth=0.5, linewidth=0.5)
plt.savefig('result{}.png'.format(str(stratify)))
results.to_csv('results.csv')