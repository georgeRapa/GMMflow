import torch, torchsde
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Independent, Categorical, MixtureSameFamily
import ot
class GMMflow(nn.Module):
    def __init__(self, Mu0, Mu1, S0, S1, W0=None, W1=None, Lambda=None, epsilon=0.5, device='cpu'):
        super().__init__()

        # Main class for solving the GMM SB with trivial prior dynamics
        # dX = u_t(X_t) dt + sqrt(epsilon) dW
        # N0: components of initial mixture
        # N1: components of terminal mixture
        # n: state dimensions
        # Mu0, Mu1: Initial/Terminal means, dimensions N0 x n, N1 x n
        # S0,  S1:  Initial/Terminal diagonal covariances, dimensions N0 x n, N1 x n
        # W0: Initial GMM weights. Must sum to 1
        # W1: Final GMM weights
        # epsilon: scalar diffusion term >= 0.

        self.device = device

        if torch.is_tensor(Mu0):
            self.Mu0 = Mu0
            self.Mu1 = Mu1
            self.S0 = S0
            self.S1 = S1
        else:
            self.Mu0 = torch.tensor(Mu0, device=device) # Initial means
            self.Mu1 = torch.tensor(Mu1, device=device) # Terminal means
            self.S0 = torch.tensor(S0, device=device) # Initial Covariances
            self.S1 = torch.tensor(S1, device=device) # Terminal Covariances

        self.N0 = Mu0.shape[0]  # number of initial Gaussians
        self.N1 = Mu1.shape[0]  # number of final Gaussians

        if W0 is None:
            self.W0 = torch.ones(self.N0, device=device)/self.N0
        else:
            self.W0 = W0

        if W1 is None:
            self.W1 = torch.ones(self.N1, device=device)/self.N1
        else:
            self.W1 = W1

        self.n = Mu0.shape[1]  # state dimension

        self.epsilon = epsilon
        self.noise_type = "additive"
        self.sde_type = "ito"

        # Evaluate some functions for the analytic calculation of the drift term:

        self.Ds = torch.sqrt(4 * self.S0.unsqueeze(1).repeat(1, self.N1, 1) * self.S1.repeat(self.N0, 1, 1) + epsilon ** 4)
        self.Cs = 0.5 * (self.Ds - self.epsilon ** 2)

        self.mu =  lambda t: ((1 - t) * self.Mu0.unsqueeze(1).repeat(1, self.N1, 1)
                              + t * self.Mu1.repeat(self.N0, 1, 1))

        self.Sigma = lambda t: ((1-t) ** 2 * self.S0.unsqueeze(1).repeat(1, self.N1, 1) + t ** 2 * self.S1.repeat(self.N0, 1, 1)
                                + 2 * t * (1 - t) * self.Cs
                                + self.epsilon ** 2 * t * (1 - t))

        self.dSigma = lambda t:  (2 * (t - 1) * self.S0.unsqueeze(1).repeat(1, self.N1, 1)
                               + 2 * t * self.S1.repeat(self.N0, 1, 1)
                               + 2 * (1 - 2 * t) * self.Cs
                               + self.epsilon ** 2 * (1 - 2 * t))

        self.Pt = lambda t: (t * self.S1.repeat(self.N0, 1, 1)
                          + (1 - t) * self.Cs)

        self.Qt = lambda t: ((1 - t) * self.S0.unsqueeze(1).repeat(1, self.N1, 1)
                             + t * self.Cs)

        self.St = lambda t: self.Pt(t) - self.Qt(t) - self.epsilon ** 2 * t

        self.v = self.Mu1.repeat(self.N0, 1, 1) - self.Mu0.unsqueeze(1).repeat(1, self.N1, 1)

        if Lambda is not None:
            self.Lambda = Lambda
        else:
            self.calc_coupling()
    def calc_coupling(self):

        #  Build component level transport cost by numerically integrating equation (7a)
        #  for the optimal policy given by equations (8), (10a)-(10e).

        T = torch.linspace(0, 0.99, 100, device=self.device)[:, None, None, None]

        self.C = (0.01 * (self.St(T) ** 2 / self.Sigma(T)).sum(dim=0).sum(dim=-1)
             + torch.linalg.vector_norm(self.v, dim=-1)**2) # component level transport cost.

        self.Lambda = ot.emd(self.W0,
                             self.W1,
                             self.C) #ot.dist(self.Mu0, self.Mu1) works also, if all the covariances are the same.
    def calc_u(self, X, t):
        # Calculate velocity field from conditional policies and the component-level transport plan.
        # Parallelized with respect to Batch and component dimensions.
        # B batch size
        # n: state dimension
        # X: state to evaluate policy (B, n)
        # t: common time for all entries in X

        B = X.shape[0]

        K = (self.St(t) / self.Sigma(t)).expand(B, -1, -1, -1) # optimal gains, dimensions B x N0 x N1 x n
        Mut = self.mu(t).expand(B, -1, -1, -1) # optimal means, dimensions B x N0 x N1 x n

        X = X[:, None, None, :].expand(-1, self.N0, self.N1, -1) # current state, dimensions B x N0 x N1 x n

        U = K * (X - Mut) + self.v.expand(B, -1, -1, -1) # conditional policies, dimensions B x N0 x N1 x n

        W = Independent(Normal(loc=Mut,
                               scale=torch.sqrt(self.Sigma(t))),1).log_prob(X) # Go with the flow weights B x N0 x N1 (log scale)

        W = W + np.log(2*np.pi) * self.n/2 # multiplying every w does not change the result; improved numerics
        W = torch.clip(W, -50, 50) # avoids numerical stability issues with very small exponents
        W = torch.exp(W)*self.Lambda.expand(B, -1, -1)
        W = W.unsqueeze(-1).expand(-1, -1, -1, self.n) # final weights after taking transport plan into account B x N0 x N1 x n 

        u = (U * W).sum(dim=1).sum(dim=1)

        return u/W.sum(dim=1).sum(dim=1)

    def f(self, t, y):
        return self.calc_u(y, t)

    def g(self, t, y):
        return self.epsilon * torch.eye(self.n, device=self.device).repeat(y.shape[0], 1, 1)

    def calc_expJ(self):
        # calculate the upper bound from equation (16b) for the transport cost.

        return (self.Lambda * self.C).sum()

    def calc_rho(self, t):

        # Calculate the state distribution at time t without integration

        mix = Categorical(self.Lambda.reshape(-1))
        rho = MixtureSameFamily(mix, Independent(Normal(loc=self.Mu1 * t + (1-t) * self.Mu0,
                                                        scale=self.Sigma(t)), 1))
        return rho

    def sample_rho(self, t, B):
        # Create B samples from the distribution of the state
        # at time t without simulating the dynamics.
        self.calc_rho(t).sample([B])
    def calc_J(self, B=5000):
        # calculate the true transport cost of the policy

        J = torch.tensor(0., device=self.device)
        T = torch.linspace(0, 0.99, 100, device=self.device)
        for t in T:
            J += 0.01* torch.mean(torch.linalg.vector_norm(self.calc_u(self.sample_rho(t, B), t), dim=1)**2, dim=0)

        return J