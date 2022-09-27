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

def stratify_sample(tab, size=1, strat_cols=(0)):
    '''

    This function takes a list/iterable (tab), and returns a random
    sample from its components, of length "size". Samples will be
    selected with a probability inversely proportional to their
    population in tab, according to the values in column index
    "strat_cols".

    @param tab: iterable object containing values
    @param size: Int. desired size of output sample
    @param strat_cols: tuple of ints. Indices of columns with tab
        to stratify the sample using.
    @return: list. stratified sample
    '''

    # join columns indicated in strat_cols and convert to single string
    all = []
    for col in strat_cols:
        strat_data = pd.DataFrame([s[col] for s in tab])
        # need to round continuous numbers to reduce sparsity - this can be tweaked
        strat_data = strat_data.round(decimals=3)
        strat_data = strat_data.astype('str')
        all = all + [strat_data.loc[:, i] for i in range(strat_data.shape[1])]
    strat_data = pd.concat(all, axis=1)
    strat_data['comp'] = strat_data.apply(lambda x: ' | '.join(x), axis=1)
    strat_data = strat_data.loc[:, ['comp']]

    # count populations, and generate sampling probabilities
    counts = strat_data['comp'].value_counts(normalize=False).reindex(strat_data['comp'])
    probs_to_sample = (1./counts) / (1./counts).sum()
    indices = np.random.choice(len(tab), size=size, p=probs_to_sample.to_list())
    return [tab[i] for i in indices]


def pad_action(act, act_param):
    '''


    @param act:
    @param act_param:
    @return:
    '''
    N = len(act_param)
    params = [np.zeros((N,), dtype=np.float32), np.zeros((N,), dtype=np.float32), np.zeros((N,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


class Actor(nn.Module):
    def __init__(self, state_size, action_param_size, action_size,
                 hidden_layers=(128,), activation=nn.ReLU,
                 l2=0., lr=1e-3,
                 verbose=None, dropout=None, random_state=1,
                 device="CPU",
                 output_layer_init_std=1e-3):
        '''

        :param layers: list of integers representing sizes of hidden layers.
            ie: [5,4] will give a NN with 2 hidden layers, first hidden layer of 5 nodes,
            and second hidden layer of 4 nodes
        :param activationfunc: default tanh. activation function to use throughout the network
        :param l2: float. l2 penalty parameter
        :param lr: float. learning rate
        :param random_state: seed for pytorch
        :param verbose: None, 'v' or 'vv' determining level of verbosity
        :param dropout: None, or list (same dims as layers) corresponding to dropout factor
            at each layer
        :param device: device for torch to put model on
        '''

        super(Actor, self).__init__()

        torch.manual_seed(random_state)

        if dropout is not None:
            assert isinstance(dropout, list)
            assert len(dropout) == len(hidden_layers)

        self.activation = activation
        self.hidden_layers = hidden_layers
        self.l2 = l2
        self.lr = lr
        self.verbose = verbose
        self.dropout = dropout
        self.criterion = nn.MSELoss()

        # Build input layer
        sequentials = []
        nInputs = state_size + action_param_size
        sequentials.append(nn.Linear(nInputs, self.hidden_layers[0]).to(device))

        # Build hidden layers
        for i in range(len(self.hidden_layers) - 1):
            sequentials.append(self.activation().to(device))
            if self.dropout is not None: sequentials.append(nn.Dropout(self.dropout[i]))
            layer = nn.Linear(self.hidden_layers[i],
                                         self.hidden_layers[i + 1]).to(device)
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity=activation)
            sequentials.append(layer)


        # Build output layer
        sequentials.append(self.activation().to(device))
        out_layer = nn.Linear(self.hidden_layers[-1], action_size)
        torch.nn.init.normal_(out_layer.weight, std=output_layer_init_std)
        sequentials.append(out_layer)
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


class ParamNet(nn.Module):
    def __init__(self, state_size, action_param_size,
                 hidden_layers=(128,), activation=nn.ReLU,
                 l2=0., lr=1e-4,
                 verbose=None, dropout=None, random_state=1,
                 device="CPU",
                 output_layer_init_std=1e-3):

        super(ParamNet, self).__init__()

        torch.manual_seed(random_state)

        if dropout is not None:
            assert isinstance(dropout, list)
            assert len(dropout) == len(hidden_layers)

        self.activation = activation
        self.hidden_layers = hidden_layers
        self.l2 = l2
        self.lr = lr
        self.verbose = verbose
        self.dropout = dropout

        # Build input layer
        sequentials = []
        nInputs = state_size
        sequentials.append(nn.Linear(nInputs, self.hidden_layers[0]).to(device))

        # Build hidden layers
        for i in range(len(self.hidden_layers) - 1):
            sequentials.append(self.activation().to(device))

            if self.dropout is not None: sequentials.append(nn.Dropout(self.dropout[i]))
            layer = nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]).to(device)
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity=activation)
            torch.nn.init.zeros_(layer.bias)
            sequentials.append(layer)

        # Build output layer
        sequentials.append(self.activation().to(device))
        out_layer = nn.Linear(self.hidden_layers[-1], action_param_size)
        torch.nn.init.normal_(out_layer.weight, std=output_layer_init_std)
        sequentials.append(out_layer)
        self.stack = nn.Sequential(*sequentials).to(device)
        self.device = device

        self.optimizer = optim.Adam(self.parameters(),
                                    lr=lr,
                                    weight_decay=l2)

    def forward(self, X):
        logits = self.stack(X)
        return logits

    def fit_batch(self, states, action_params, clipping=None):
        from copy import deepcopy

        delta_a = deepcopy(action_params.grad.data)

        action_params = self(states)
        out = -torch.mul(delta_a, action_params)
        self.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if clipping:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clipping)
        self.optimizer.step()
        return


class PDQNAgent:
    def __init__(self, observation_space, action_space,
                 actorNet_kwargs={},
                 paramNet_kwargs={},
                 memory_size=10000,
                 gamma=0.9,
                 epsilon_start=1.0,
                 epsilon_min=0.05,
                 epsilon_bumps=[], # when epsilon hits these values, reset to original epsilon
                 epsilon_decay=0.999,
                 batch_size=128,
                 train_start=500,
                 action_param_lims=None,
                 grad_clipping=10.,
                 stratify_replay_memory=True,
                 actor_softness=0.1,
                 param_softness=0.01,
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        self.state_size = observation_space.spaces[0].shape[0]
        self.action_size = action_space.spaces[0].n
        action_param_sizes = np.array(
            [action_space.spaces[1].spaces[i].shape[0] for i in range(self.action_size)])
        self.action_param_size = int(action_param_sizes.sum())

        self.actor_softness = actor_softness
        self.param_softness = param_softness
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon_start  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_init = float(epsilon_start)
        self.epsilon_bumps = epsilon_bumps
        self.batch_size = batch_size
        self.train_start = train_start
        self.device = device
        self.clipping = grad_clipping
        self.stratify_replay_memory = stratify_replay_memory
        if not action_param_lims:
            self.action_param_lims = np.array([(-1, 1) for i in range(self.action_size)])  # default

        # format the network params
        actorNet_kwargs['device'] = device  # ensure everything is on same device
        actorNet_kwargs['action_size'] = self.action_size
        actorNet_kwargs['action_param_size'] = self.action_param_size
        actorNet_kwargs['state_size'] = self.state_size
        paramNet_kwargs['device'] = device
        paramNet_kwargs['action_param_size'] = self.action_param_size
        paramNet_kwargs['state_size'] = self.state_size

        # build networks
        self.actorNet = Actor(**actorNet_kwargs).double()
        self.paramNet = ParamNet(**paramNet_kwargs).double()

        # create duplicates of the models (target models)
        self.actor_dupe = deepcopy(self.actorNet)
        self.param_dupe = deepcopy(self.paramNet)

    def remember(self, state, action, action_param, reward, next_state, done):
        '''

        Appends state/action/action_param etc as an item in the agents recall memory
        '''
        self.memory.append((state, action, action_param, reward, next_state, done))

    def act(self, state):
        '''

        @param state: state vector
        @return: action index, action_param value corresponding to action, full action_params from
        paramNet.
        '''
        if random.uniform(0, 1) < self.epsilon:  # implement epsilon exploration
            action = np.random.randint(0, self.action_size)
            action_params = torch.from_numpy(
                np.array([np.random.uniform(low, high) for low, high in self.action_param_lims]))
            low, high = self.action_param_lims[action]
            ap = np.random.uniform(low, high)
        else:
            state = torch.from_numpy(state).to(self.device)
            action_params = self.paramNet(state)
            concat_state = torch.cat((state, action_params), dim=0)
            Q = self.actorNet(concat_state)
            action = np.argmax(Q.detach().cpu().numpy())
            ap = float(action_params[action])
        return action, [ap], action_params.detach().cpu()

    def replay(self):
        '''

        Draws a sample from recall memory, and trains the agent by one iteration.
        '''

        if len(self.memory) < self.train_start:
            return

        if len(self.epsilon_bumps) > 0:
            if self.epsilon <= self.epsilon_bumps[0]:
                self.epsilon = self.epsilon_init
                self.epsilon_bumps.pop(0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Randomly sample minibatch from the memory
        if self.stratify_replay_memory:
            minibatch = stratify_sample(self.memory,
                                        size=self.batch_size,
                                        strat_cols=(0, 1)) # stratify by cols 0 (state) and 1 (action)
        else:
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        # Train the actor network
        states = torch.from_numpy(np.array([s[0] for s in minibatch])).to(self.device)
        next_states = torch.from_numpy(np.array([s[4] for s in minibatch])).to(self.device)
        dones = torch.from_numpy(np.array([s[5] for s in minibatch]).astype('int')).to(self.device)
        rewards = torch.from_numpy(np.array([s[3] for s in minibatch])).to(self.device)
        actions = torch.from_numpy(np.array([s[1] for s in minibatch])).to(self.device)
        # Do we pull action params here and use in place of action_params directly below?
        action_params = torch.concat([s[2].reshape(-1, self.action_size) for s in minibatch], dim=0).to(self.device)

        with torch.no_grad():
            next_action_param = self.param_dupe(next_states).detach()
            next_actor_inputs = torch.cat((next_states, next_action_param), dim=1)

            actor_inputs = torch.cat((states, action_params), dim=1)
            Q = self.actor_dupe(actor_inputs).detach()
            Q_next = self.actor_dupe(next_actor_inputs).detach()
            Q_target_next = torch.max(Q_next, dim=1)[0]
            not_complete = (1 - dones)

            target = deepcopy(Q)
            target_flat = rewards + self.gamma * Q_target_next * not_complete
            indices = [(i, int(actions[i])) for i in range(len(actions))]
            for (i, j) in indices:
                target[i, j] = target_flat[i]

        self.actorNet.fit_batch(actor_inputs, target, clipping=self.clipping)

        # Secondly, train the paramNet
        with torch.no_grad():
            action_params = self.paramNet(states)
        action_params.requires_grad = True
        actor_inputs = torch.cat((states, action_params), dim=1)
        Qs = self.actorNet(actor_inputs)
        Q_loss = torch.mean(torch.sum(Qs, 1)) # Goal is to maximise Q
        self.actorNet.zero_grad()
        Q_loss.backward()

        delta_a = deepcopy(action_params.grad.data)
        action_params = self.paramNet(states)
        # shift gradients in paramNet so parameters are bound by limits
        max_params = torch.from_numpy(self.action_param_lims[:, 1]).to(self.device)
        min_params = torch.from_numpy(self.action_param_lims[:, 0]).to(self.device)
        geq = (delta_a > 0)
        delta_a[geq] *= max_params.sub(action_params).div(max_params - min_params)[geq]
        delta_a[~geq] *= action_params.sub(min_params).div(max_params - min_params)[~geq]

        out = -torch.mul(delta_a, action_params)
        self.paramNet.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if self.clipping:
            torch.nn.utils.clip_grad_norm_(self.paramNet.parameters(), self.clipping)
        self.paramNet.optimizer.step()

        # implement soft update for training stability
        for dupe_param, param in zip(self.actor_dupe.parameters(), self.actorNet.parameters()):
            dupe_param.data.copy_(self.actor_softness* param.data + (1.0 - self.actor_softness) * dupe_param.data)
        for dupe_param, param in zip(self.param_dupe.parameters(), self.paramNet.parameters()):
            dupe_param.data.copy_(self.param_softness * param.data + (1.0 - self.param_softness) * dupe_param.data)

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
        self.actorNet.load_state_dict(torch.load(actorNet))
        self.paramNet.load_state_dict(torch.load(paramNet))

        # create duplicates of the models (target networks)
        self.actor_dupe = deepcopy(self.actorNet)
        self.param_dupe = deepcopy(self.paramNet)


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
    for e in range(episodes):
        state, _ = env.reset()
        done = False
        tot_reward = 0
        while not done:

            action, action_param, all_action_params = agent.act(state)

            formatted_params = [np.zeros((agent.action_param_size,), dtype=np.float32)]*agent.action_size
            formatted_params[action][:] = action_param

            (next_state, _), reward, done, _ = env.step((action, formatted_params))

            #if done: # make it bad to die
            #    reward = reward - 1e-2
            #reward = reward*100 # scale it up to something reasonable

            if train:
                agent.remember(state, action, all_action_params, reward, next_state, done)
                agent.replay()

            state = next_state
            tot_reward += reward
            if (e % 100 == 0 and render) or ~train:
                env.render()

            if done:
                episode_scores.append(tot_reward)

            if done and e >= 100 and e % 100 == 0:
                dateTimeObj = datetime.now()
                timestampStr = dateTimeObj.strftime("%H:%M:%S")
                last_scores = episode_scores[-100:]

                print("episode: {}/{}, score ave {:.3} range: {:.3}-{:.3}, e: {:.2}, time: {}".format(e, episodes,
                                                                                      np.mean(last_scores),
                                                                                      min(last_scores),
                                                                                      max(last_scores),
                                                                                      agent.epsilon,
                                                                                      timestampStr))

    return episode_scores
