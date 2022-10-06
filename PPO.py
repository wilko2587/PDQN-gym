import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from datetime import datetime
from copy import deepcopy
import pandas as pd
import os

''' # relevant papers
https://arxiv.org/abs/1905.04388
https://arxiv.org/abs/1810.06394
https://arxiv.org/abs/1509.01644
'''


class FFnet(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_layers=(128,), activation=nn.ReLU,
                 l2=0., lr=1e-3,
                 dropout=None, random_state=1,
                 device="CPU",
                 layer_std=None,
                 output_function=None):
        '''

        @param: input_size: int. number of inputs to model.
        @param: output_size. int. number of output nodes
        @param: hidden_layers. tuple. contains integers dictating the number of nodes in the
            network's hidden layers
        @param: activation: activation function
        @param: l2: float. l2 regularisation constant
        @param: lr: float. learning rate
        @param: dropout: tuple. contains integers dictating dropout rate for each hidden layer.
            must be same dimension as hidden_layers.
        @param: device: torch device to put the model on
        @param: output_layer_init_std: stdev of initial weights in output layer.
        @param: output_function: function to apply after final layer of network.
        '''

        super(FFnet, self).__init__()

        torch.manual_seed(random_state)

        if dropout is not None:
            assert isinstance(dropout, list)
            assert len(dropout) == len(hidden_layers)

        self.activation = activation
        self.hidden_layers = hidden_layers
        self.l2 = l2
        self.lr = lr
        self.dropout = dropout
        self.criterion = nn.MSELoss()

        if layer_std is None:
            layer_std = 1.

        # Build input layer
        sequentials = []
        layer = nn.Linear(input_size, self.hidden_layers[0], dtype=torch.double).to(device)
        torch.nn.init.uniform_(layer.weight, -layer_std/np.sqrt(layer.weight.size(1)), layer_std/np.sqrt(layer.weight.size(1)))
        sequentials.append(layer)

        # Build hidden layers
        for i in range(len(self.hidden_layers) - 1):
            sequentials.append(self.activation().to(device))
            if self.dropout is not None: sequentials.append(nn.Dropout(self.dropout[i]))
            layer = nn.Linear(self.hidden_layers[i],
                                         self.hidden_layers[i + 1], dtype=torch.double).to(device)
            torch.nn.init.uniform_(layer.weight, -layer_std / np.sqrt(layer.weight.size(1)), layer_std/np.sqrt(layer.weight.size(1)))
            #nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            sequentials.append(layer)

        # Build output layer
        sequentials.append(self.activation().to(device))
        out_layer = nn.Linear(self.hidden_layers[-1], output_size, dtype=torch.double)
        torch.nn.init.uniform_(out_layer.weight, -layer_std / np.sqrt(out_layer.weight.size(1)), layer_std/np.sqrt(out_layer.weight.size(1)))
        sequentials.append(out_layer)
        if str(output_function).lower() == 'tanh':
            sequentials.append(nn.Tanh().to(device))
        if str(output_function).lower() == 'softmax':
            sequentials.append(nn.Softmax(dim=0).to(device))

        self.stack = nn.Sequential(*sequentials).to(device)
        self.device = device

        self.optimizer = optim.Adam(self.parameters(),
                                    lr=lr,
                                    weight_decay=l2)

    def forward(self, X):
        logits = self.stack(X)
        return logits

    def fit_batch(self, X, y, clipping=None):

        X, y = X.to(self.device), y.to(self.device)  # Send to GPU
        self.optimizer.zero_grad()
        logits = self(X)

        tloss = self.criterion(logits, y)
        tloss.backward()
        if clipping:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clipping)

        self.optimizer.step()


class PPOAgent:
    def __init__(self, observation_space, action_space,
                 qNet_kwargs={},
                 policyNet_kwargs={},
                 paramNet_kwargs={},
                 gamma=0.9,
                 batch_size=128,
                 action_param_lims=None,
                 grad_clipping=10.,
                 qSoftness=0.1,
                 policySoftness=0.1,
                 paramSoftness=0.01,
                 param_gradient_method='constrained',
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        """

        @param: observation_space: gym environment observation_space object
        @param: action_sapce: gym environment action_space
        @param: actorNet_kwargs: initialisation parameters for actor network
        @param: paramNet_kwargs: initialisation parameters for param network
        @param: memory_size: size of the agent's replay memory
        @param: gamma: discount rate applied to future rewards
        @param: epsilon_start: float. 0->1. Initial value for epsilon in epsilon greedy.
        @param: epsilon_min: float. Minimum value for epsilon
        @param: epsilon_bumps: list. When epsilon decays to a value in this list, it is reset
            to its initial value. The value is then removed from the list.
        @param: epsilon_grad: float. Artificial increase in epsilon for each unit of reward
            beyond that which the agent has received on average.
        @param: epsilon_decay. Exponential decay rate for epsilon.
        @param: batch_size: number of items to read from recall memory on each step.
        @param: train_start: minimum number of samples in memory before training starts
        @param: action_param_lims. Hard limits on the values allowed for action parameters,
            If None, defaults to range of -1 -> +1 for all action parameters.
        @param: grad_clipping: value of which to clip gradients.
        @param: noise_level: decimal std of noise to apply to param net exploration
        @param: stratify_replay_memory: If True, bot will use a stratified method to sample the
            memory to increase the prevailance of rare state/action pairs. This can run slow.
        @param: actor_softness: softness parameter applied to the actor updates (range 0->1)
        @param_gradient_method: 'unconstrained' or 'constrained'. If constrained, uses tanh to compress param outputs
        @param: param_softness: softness parameter applied to param update (range 0->1)
        """
        self.state_size = observation_space.spaces[0].shape[0]
        self.action_size = action_space.spaces[0].n
        action_param_sizes = np.array(
            [action_space.spaces[1].spaces[i].shape[0] for i in range(self.action_size)])
        self.action_param_size = int(action_param_sizes.sum())
        self.actions = np.array(np.arange(0, self.action_size))

        self.qSoftness= qSoftness
        self.paramSoftness = paramSoftness
        self.policySoftness = policySoftness
        self.gamma = gamma  # discount rate
        self.batch_size = batch_size
        self.device = device
        self.clipping = grad_clipping
        self.param_gradient_method = param_gradient_method
        if not action_param_lims:
            self.action_param_lims = np.array([(-1, 1) for i in range(self.action_size)])  # default

        # format the network params
        qNet_kwargs['device'] = device  # ensure everything is on same device
        qNet_kwargs['input_size'] = self.state_size + self.action_param_size
        qNet_kwargs['output_size'] = 1
        policyNet_kwargs['device'] = device
        policyNet_kwargs['input_size'] = self.state_size
        policyNet_kwargs['output_size'] = self.action_size
        policyNet_kwargs['output_function'] = "softmax"
        paramNet_kwargs['device'] = device
        paramNet_kwargs['input_size'] = self.state_size
        paramNet_kwargs['output_size'] = self.action_param_size
        paramNet_kwargs['layer_std'] = 1.

        if param_gradient_method == 'constrained':
            paramNet_kwargs['output_function'] = 'tanh'
        else:
            paramNet_kwargs['output_function'] = None

        # build networks
        self.qNet = FFnet(**qNet_kwargs).double()
        self.policyNet = FFnet(**policyNet_kwargs).double()
        self.paramNet = FFnet(**paramNet_kwargs).double()

        # create duplicates of the models (target models)
        self.q_dupe = deepcopy(self.qNet)
        self.policy_dupe = deepcopy(self.policyNet)
        self.param_dupe = deepcopy(self.paramNet)

    def act(self, state):
        '''

        @param state: state vector
        @return: action index, action_param value corresponding to action, full action_params from
        paramNet.
        '''
        with torch.no_grad():
            state = torch.DoubleTensor(state).to(self.device)
            action_probs = self.policyNet(state).detach()
            action = np.random.choice(self.actions, p=action_probs)
            action_params = self.paramNet(state).detach()
            ap = action_params[action]
            return action, ap, action_params

    def update_networks(self, states=None,
                        actions=None, rewards=None):

        assert len(rewards)==len(states)==len(actions)

        self.paramNet.zero_grad()
        self.qNet.zero_grad()
        self.policyNet.zero_grad()
        self.param_dupe.zero_grad()
        self.q_dupe.zero_grad()
        self.policy_dupe.zero_grad()

        states = states.type(torch.DoubleTensor)
        rewards = rewards.type(torch.DoubleTensor)
        actions = actions.type(torch.LongTensor)

        # 1) update policy network under static Q network
        with torch.no_grad():
            discount = torch.Tensor(np.array([self.gamma**i for i in range(len(rewards))]))
            Qvals = [torch.sum(rewards[i:]*discount[i:])/discount[i] for i in range(len(rewards))]
            Qvals = torch.Tensor(np.array(Qvals)).double()
            action_params = self.param_dupe(states).detach() # soft
            Qpreds = self.q_dupe(torch.cat([states, action_params], dim=1)).detach() # soft
            ads = Qpreds - Qvals.unsqueeze(dim=1)

        action_dummies = torch.nn.functional.one_hot(actions, num_classes=self.action_size)
        loss = torch.sum(self.policyNet(states)*action_dummies*ads)

        self.policyNet.zero_grad()
        loss.backward()
        self.policyNet.optimizer.step()

        # 2) update Q network
        action_params = self.param_dupe(states) # soft
        qnet_inputs = torch.cat((states, action_params), dim=1)
        Qs = self.qNet(qnet_inputs).squeeze(dim=1)
        loss = torch.nn.MSELoss()(Qs, Qvals)
        self.qNet.zero_grad()
        loss.backward()
        self.qNet.optimizer.step()

        # 3) update paramnet on the orig q network
        action_params = self.paramNet(states)
        qnet_inputs = torch.cat((states, action_params), dim=1)
        Qs = self.q_dupe(qnet_inputs)
        loss = -torch.sum(Qs) # want to maximize Q
        self.paramNet.zero_grad()
        loss.backward()
        self.paramNet.optimizer.step()

        self.paramNet.zero_grad()
        self.qNet.zero_grad()
        self.policyNet.zero_grad()
        self.param_dupe.zero_grad()
        self.q_dupe.zero_grad()
        self.policy_dupe.zero_grad()

        # implement soft update for training stability
        for dupe_param, param in zip(self.q_dupe.parameters(), self.qNet.parameters()):
            dupe_param.data.copy_(self.qSoftness* param.data + (1.0 - self.qSoftness) * dupe_param.data)
        for dupe_param, param in zip(self.param_dupe.parameters(), self.paramNet.parameters()):
            dupe_param.data.copy_(self.paramSoftness * param.data + (1.0 - self.paramSoftness) * dupe_param.data)
        for dupe_param, param in zip(self.policy_dupe.parameters(), self.policyNet.parameters()):
            dupe_param.data.copy_(self.policySoftness * param.data + (1.0 - self.policySoftness) * dupe_param.data)
        return

    def save(self, path='./models', id=''):
        '''

        @param path: path to save to
        @param id: string/int. ID to save models with
        '''
        torch.save(self.actorNet.state_dict(), os.path.join(path, 'actorNet_id{}.pt'.format(id)))
        torch.save(self.paramNet.state_dict(), os.path.join(path, 'paramNet_id{}.pt'.format(id)))

    def load(self, path='./models', id=''):
        '''

        @param path: path to load models from
        @param id: id of model
        '''

        # load the models
        actorNet = os.path.join(path, 'actorNet_id{}.pt'.format(id))
        paramNet = os.path.join(path, 'paramNet_id{}.pt'.format(id))
        self.actorNet.load_state_dict(torch.load(actorNet, map_location=self.device))
        self.paramNet.load_state_dict(torch.load(paramNet, map_location=self.device))

        # create duplicates of the models (target networks)
        self.actor_dupe = deepcopy(self.actorNet)
        self.param_dupe = deepcopy(self.paramNet)
        return


def play(env, agent, episodes=1000, render=True,
                seed=1, train=True):
    """

    @param env: gym environment for agent to use
    @param agent: RL agent (suited for env)
    @param episodes: int. number of episodes to train for
    @param render: bool. True will render pygame window every 100 episodes.
    @param seed: int. seed for all the non-deterministic modules.
    @param train: bool. If true, agent will train as it plays
    @return: list. final scores (sum of rewards) received by bot. One element per episode.
    """

    # seed
    random.seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    episode_scores = []
    nactions = []
    for e in range(episodes):
        state, _ = env.reset()
        done = False
        tot_reward = 0
        states = []
        rewards = []
        actions = []
        while not done:

            action, action_param, all_action_params = agent.act(state)


            formatted_params = all_action_params.numpy().reshape([agent.action_param_size, 1])

            (next_state, _), reward, done, _ = env.step((action, formatted_params))

            #if e%100==0 and len(states)==0:
            #    print(len(states), ' ',state)
            #    print(formatted_params)
            #    print('--')
            #if e%100==0 and len(states)==1:
            #    print(len(states), ' ',state)
            #    print(formatted_params)
            #    print('--')
            #if e % 100 == 0 and len(states) == 2:
            #    print(len(states),' ', state)
            #    print(formatted_params)
            #    print('--')

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            tot_reward += reward
            episode_scores.append(tot_reward)
            if done:
                agent.update_networks(states=torch.Tensor(np.array(states)),
                                      actions=torch.Tensor(np.array(actions)),
                                      rewards=torch.Tensor(np.array(rewards)))

            if (e % 100 == 0 and render) or (not train and render):
                env.render()

            if done and e >= 100 and e % 100 == 0:
                dateTimeObj = datetime.now()
                timestampStr = dateTimeObj.strftime("%H:%M:%S")
                last_scores = episode_scores[-100:]

                print("episode: {}/{}, score ave {:.3} range: {:.3}-{:.3}, time: {}".format(e, episodes,
                                                                                      np.mean(last_scores),
                                                                                      min(last_scores),
                                                                                      max(last_scores),
                                                                                      timestampStr))

    return episode_scores
