from PPO import PPOAgent, play
import gym
import gym_platform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__=='__main__':
    env = gym.make("Platform-v0")

    # Network/setup params
    qNet_kwargs = {'hidden_layers': (128, 128), 'l2': 1e-6, 'lr': 1e-4}
    policyNet_kwargs = {'hidden_layers': (128, 128), 'l2': 1e-6, 'lr': 1e-4}
    paramNet_kwargs = {'hidden_layers': (128, 128), 'l2': 1e-6, 'lr': 1e-5}
    Nepisodes = 20000

    # initialise PDQN agent
    agent = PPOAgent(observation_space=env.observation_space,
                          action_space=env.action_space,
                          qNet_kwargs=qNet_kwargs,
                          policyNet_kwargs=policyNet_kwargs,
                          paramNet_kwargs=paramNet_kwargs,
                          batch_size=128,
                          gamma=0.9,
                          grad_clipping=10.,
                          qSoftness=1.,
                          policySoftness=1.,
                          paramSoftness=1.)

    #agent.load(id=1)
    # train agent, and get scores for each episode
    scores = play(env, agent, episodes=Nepisodes, render=False, train=True)

    #agent.save(id=1)

    # ----- Plotting -----
    # bin the episodes into 500 length bins
    scores_binned = pd.DataFrame(index=np.floor(np.arange(0, len(scores)) / 500.) * 500, columns=['score'], data=scores)
    scores_binned = scores_binned.reset_index()
    scores_binned = scores_binned.rename(columns={'index': 'episode'})
    f = plt.figure()
    sns.pointplot(data=scores_binned, y='score', x='episode', errwidth=0.5, linewidth=0.5)

    # save results
    plt.savefig('result.png')
    scores_binned.to_csv('results.csv')
