# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False # No, we use CPU.
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
class DQNNetwork:
    """
    A class representing a deep Q network.
    Note that this function is batched. I.e. ``s`` is assumed to be a numpy array of dimension ``batch_size x n``

    The following example shows how you can evaluate the Q-values in a given state. An example:

    .. runblock:: pycon

        >>> from irlc.ex13.torch_networks import TorchNetwork
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> env = gym.make("CartPole-v1")
        >>> Q = TorchNetwork(env, trainable=True, learning_rate=0.001) # DQN network requires an env to set network dimensions
        >>> batch_size = 32 # As an example
        >>> states = np.random.rand(batch_size, env.observation_space.shape[0]) # Creates some dummy input
        >>> states.shape    # batch_size x n
        >>> qvals = Q(states) # Evaluate Q(s,a)
        >>> qvals.shape # This is a tensor of dimension batch_size x actions
        >>> print(qvals[0,1]) # Get Q(s_0, 1)
        >>> Y = np.random.rand(batch_size, env.action_space.n) # Generate target Q-values (training data)
        >>> Q.fit(states, Y)                      # Train the Q-network for 1 gradient descent step
    """
    def update_Phi(self, source, tau=0.01):
        r"""
        Update (adapts) the weights in this network towards those in source by a small amount.

        For each weight :math:`w_i` in (this) network, and each corresponding weight :math:`w'_i` in the ``source`` network,
        the following Polyak update is performed:

        .. math::
            w_i \leftarrow w_i + \tau (w'_i - w_i)

        :param source: Target network to update towards
        :param tau: Update rate (rate of change :math:`\\tau`
        :return: ``None``
        """

        raise NotImplementedError

    def __call__(self, s):
        """
        Evaluate the Q-values in the given (batched) state.

        :param s: A matrix of size ``batch_size x n`` where :math:`n` is the state dimension.
        :return: The Q-values as a ``batch_size x d`` dimensional matrix where :math:`d` is the number of actions.
        """
        raise NotImplementedError

    def fit(self, s, target): 
        r"""
        Fit the network weights by minimizing

        .. math::
            \frac{1}{B}\sum_{i=1}^B \sum_{a=1}^K \| q_\phi(s_i)_a - y_{i,a} \|^2

        where ``target`` corresponds to :math:`y` and is a ``[batch_size x actions]`` matrix of target Q-values.
        :param s: 
        :param target: 
        :return: 
        """
        raise NotImplementedError

class TorchNetwork(nn.Module,DQNNetwork):
    def __init__(self, env, trainable=True, learning_rate=0.001, hidden=30):
        nn.Module.__init__(self)
        DQNNetwork.__init__(self)
        self.env = env
        self.num_hidden = hidden
        self.num_actions = np.prod(env.action_space.shape)
        self.num_observations = np.prod(env.observation_space.shape)

    def build_model_(self):
        raise NotImplementedError

    def forward(self, s, a=None):
        raise NotImplementedError

    def __call__(self, s, a=None):
        return self.forward(s, a).detach().numpy()

    def fit(self, target, s, a=None):
        raise NotImplementedError

    def update_Phi(self, source, tau=1):
        """
        Polyak adapt weights of this class given source:
        I.e. tau=1 means adopt weights in one step,
        tau = 0.001 means adopt very slowly, tau=1 means instant overwriting
        """
        state = self.state_dict()
        for k, wa in state.items():
            wb = source.state_dict()[k]
            state[k] = wa*(1 - tau) + wb * tau
        self.load_state_dict(state)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        torch.save(self.state_dict(), path+".torchsave")

    def load(self, path):
        self.load_state_dict(torch.load(path+".torchsave"))
        self.eval() # set batch norm layers, dropout, other stuff we don't use


class TorchActorNetwork(TorchNetwork):
    def __init__(self, env, trainable=True, learning_rate=0.001, hidden=30):
        super().__init__(env, trainable, learning_rate, hidden)
        self.build_model_()
        if trainable:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if USE_CUDA:
            self.cuda()

    def build_feature_network(self):
        return (nn.Linear(self.num_observations, self.num_hidden),
                nn.ReLU(),
                nn.Linear(self.num_hidden, self.num_hidden),
                nn.ReLU())

    def build_model_(self):
        self.model = nn.Sequential(*self.build_feature_network(), nn.Linear(self.num_hidden,self.num_actions))

    def forward(self, s, a=None):
        return self.model(s) # Forward pass through the network to get action values

    def fit(self, critic, s):
        a = self.forward(s)
        loss = -critic.forward(s, a).mean()  # Get the critic's evaluation of the actions
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TorchCriticNetwork(TorchNetwork):
    def __init__(self, env, trainable=True, learning_rate=0.001, hidden=30):
        super().__init__(env, trainable, learning_rate, hidden)
        self.num_features = self.num_observations + self.num_actions  # State + Action
        self.build_model_()
        if trainable:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if USE_CUDA:
            self.cuda()

    def __call__(self, s, a):
        return self.forward(s, a).detach().numpy()

    def build_feature_network(self):
        return (nn.Linear(self.num_features, self.num_hidden),
                nn.ReLU(),
                nn.Linear(self.num_hidden, self.num_hidden),
                nn.ReLU())

    def build_model_(self):
        self.model = nn.Sequential(*self.build_feature_network(), nn.Linear(self.num_hidden,1))

    def forward(self, s, a):
        features = torch.cat((s, a), dim=1)  # Concatenate state and action
        return self.model(features)
    
    def fit(self, target, s, a):
        x = self.forward(s, a)
        loss = (torch.FloatTensor(target).detach() - x).pow(2).sum(axis=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    a = 234
    import gymnasium as gym

    env = gym.make("CartPole-v0")
    Q = DQNNetwork(env, trainable=True, learning_rate=0.001)

    # self.Q = Network(env, trainable=True)  # initialize the network
    """ Assuming s has dimension [batch_dim x d] this returns a float numpy Array
    array of Q-values of [batch_dim x actions], such that qvals[i,a] = Q(s_i,a) """
    batch_size = 32 # As an example
    # Creates some dummy input
    states = [env.reset()[0] for _ in range(batch_size)]
    states.shape    # batch_size x n

    qvals = Q(states)
    qvals.shape # This is a tensor of dimension batch_size x actions
    print(qvals[0,1]) # Get Q(s_0, 1)

    Y = np.random.rand( (batch_size, 1)) # Generate target Q-values (training data)
    Q.fit(states, Y) # Train the Q-network.



    # Q = TorchNetwork()
