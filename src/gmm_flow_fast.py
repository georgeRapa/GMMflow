import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, Independent, Categorical, MixtureSameFamily
import ot


class GMMflow_fast(nn.Module):
    """
    Main class for solving the GMM SB with trivial prior dynamics and diagonal cov. matrices.
    dx_t = u_t(x_t) dt + sqrt(epsilon) dw_t
    N0: components of initial mixture
    N1: components of terminal mixture
    n: state dimensions
    Mu0, Mu1: Initial/Terminal means, dimensions N0 x n, N1 x n
    S0,  S1:  Initial/Terminal diagonal covariances, dimensions N0 x n, N1 x n
    W0: Initial GMM weights. Must sum to 1
    W1: Final GMM weights
    epsilon: scalar diffusion term >= 0.
    Comment: Improvment over gmm_flow to exploit the sparsity of the component-level transport plan
    Lambda. For problems with many components, the speed improvement is considerable.
    """

    def __init__(self, Mu0, Mu1, S0, S1, W0=None, W1=None, Lambda=None, epsilon=0.5, device='cpu'):
        super().__init__()

        self.device = device

        self.N0 = Mu0.shape[0]
        self.N1 = Mu1.shape[0]
        self.n = Mu0.shape[1]  # state dimension

        self.Mu0 = Mu0 if torch.is_tensor(Mu0) else torch.zeros(Mu0.shape).to(self.device)
        self.Mu1 = Mu1 if torch.is_tensor(Mu1) else torch.zeros(Mu1.shape).to(self.device)
        self.S0 = S0 if torch.is_tensor(S0) else torch.zeros(S0.shape).to(self.device)
        self.S1 = S1 if torch.is_tensor(S1) else torch.zeros(S1.shape).to(self.device)
        self.W0 = torch.ones(self.N0, device=device) / self.N0 if W0 is None else W0
        self.W1 = torch.ones(self.N1, device=device) / self.N1 if W1 is None else W1

        self.epsilon = epsilon
        self.noise_type = "additive"
        self.sde_type = "ito"

        # Precompute expanded views
        S0_exp = self.S0.unsqueeze(1).expand(-1, self.N1, -1)
        S1_exp = self.S1.expand(self.N0, -1, -1)
        Mu0_exp = self.Mu0.unsqueeze(1).expand(-1, self.N1, -1)
        Mu1_exp = self.Mu1.expand(self.N0, -1, -1)

        eps2 = epsilon ** 2

        # Precompute constants
        self.Ds = torch.sqrt(4 * S0_exp * S1_exp + epsilon ** 4)
        self.Cs = 0.5 * (self.Ds - eps2)
        self.v = Mu1_exp - Mu0_exp

        # Time-dependent quantities
        self.mu = lambda t: (1 - t) * Mu0_exp + t * Mu1_exp
        self.Sigma = lambda t: (1 - t) ** 2 * S0_exp + t ** 2 * S1_exp + 2 * t * (1 - t) * (self.Cs + 0.5 * eps2)
        self.dSigma = lambda t: 2 * ((t - 1) * S0_exp + t * S1_exp + (1 - 2 * t) * (self.Cs + 0.5 * eps2))
        self.Pt = lambda t: t * S1_exp + (1 - t) * self.Cs
        self.Qt = lambda t: (1 - t) * S0_exp + t * self.Cs
        self.St = lambda t: self.Pt(t) - self.Qt(t) - eps2 * t

        if Lambda is not None:
            self._sparsify_lambda(Lambda)
        else:
            self.calc_coupling()

    def _sparsify_lambda(self, Lambda, tol=1e-10):
        """Convert Lambda matrix to sparse representation"""
        # Find non-zero entries
        nonzero_mask = Lambda > tol
        self.Lambda_indices_i = torch.nonzero(nonzero_mask, as_tuple=True)[0]
        self.Lambda_indices_j = torch.nonzero(nonzero_mask, as_tuple=True)[1]
        self.Lambda_values = Lambda[nonzero_mask]
        self.num_nonzero = len(self.Lambda_values)

        # Precompute sparse versions of expanded arrays
        i_idx = self.Lambda_indices_i
        j_idx = self.Lambda_indices_j

        # Extract only non-zero component pairs
        self.S0_sparse = self.S0[i_idx, :]  # (num_nonzero, n)
        self.S1_sparse = self.S1[j_idx, :]  # (num_nonzero, n)
        self.Mu0_sparse = self.Mu0[i_idx, :]  # (num_nonzero, n)
        self.Mu1_sparse = self.Mu1[j_idx, :]  # (num_nonzero, n)

        eps2 = self.epsilon ** 2

        # Precompute sparse constants
        self.Ds_sparse = torch.sqrt(4 * self.S0_sparse * self.S1_sparse + self.epsilon ** 4)
        self.Cs_sparse = 0.5 * (self.Ds_sparse - eps2)
        self.v_sparse = self.Mu1_sparse - self.Mu0_sparse

        # Create sparse time-dependent function handles
        self.mu_sparse = lambda t: (1 - t) * self.Mu0_sparse + t * self.Mu1_sparse
        self.Sigma_sparse = lambda t: (1 - t) ** 2 * self.S0_sparse + t ** 2 * self.S1_sparse + 2 * t * (1 - t) * (
                    self.Cs_sparse + 0.5 * eps2)
        self.Pt_sparse = lambda t: t * self.S1_sparse + (1 - t) * self.Cs_sparse
        self.Qt_sparse = lambda t: (1 - t) * self.S0_sparse + t * self.Cs_sparse
        self.St_sparse = lambda t: self.Pt_sparse(t) - self.Qt_sparse(t) - eps2 * t

    def calc_coupling(self):
        """Build component level transport transport plan"""

        S0_exp = self.S0.unsqueeze(dim=1)
        S1_exp = self.S1.unsqueeze(dim=0)

        if self.epsilon > 0.:
            I = torch.ones(self.N0, self.N1, self.n, device=self.device)
            M_eps = I + torch.sqrt(
                I + 16 / (4 * self.epsilon ** 4) * S0_exp.expand(self.N0, self.N1, -1) * S1_exp.expand(self.N0, self.N1,
                                                                                                       -1))
            # Instead of calculating the full cost, we can only use the terms that contribute.
            self.C = ot.dist(self.Mu0, self.Mu1) - self.epsilon ** 2 * (M_eps.sum(dim=-1) - M_eps.log().sum(dim=-1))

        elif self.epsilon == 0:
            # again we calculate only the terms that contribute.
            self.C = ot.dist(self.Mu0, self.Mu1) + torch.sqrt(
                S0_exp.expand(self.N0, self.N1, -1) * S1_exp.expand(self.N0, self.N1, -1)).sum(dim=-1)

        else:
            raise ValueError("Noise must be non-negative.")

        Lambda = ot.emd(self.W0, self.W1, self.C)
        self._sparsify_lambda(Lambda)

    def calc_u(self, X, t):
        # Calculate velocity field from conditional policies and the component-level transport plan.
        # Parallelized with respect to Batch and component dimensions, only using non-zero lambdas.
        # B batch size
        # n: state dimension
        # X: state to evaluate policy (B, n)
        # t: common time for all entries in X

        B = X.shape[0]

        # Use sparse function handles - only compute for non-zero lambda entries
        Sigma_t = self.Sigma_sparse(t)  # (num_nonzero, n)
        K = self.St_sparse(t) / Sigma_t  # (num_nonzero, n)
        Mut = self.mu_sparse(t)  # (num_nonzero, n)

        # Use unsqueeze for broadcasting instead of expand
        # X: (B, n) -> (B, 1, n), Mut: (num_nonzero, n) -> (1, num_nonzero, n)
        X_diff = X.unsqueeze(1) - Mut.unsqueeze(0)  # (B, num_nonzero, n)

        # Conditional policies using broadcasting
        U = K.unsqueeze(0) * X_diff + self.v_sparse.unsqueeze(0)  # (B, num_nonzero, n)

        # Calculate weights - use broadcasting for Normal distribution
        # log_prob expects (B, num_nonzero, n) and outputs (B, num_nonzero)
        W = Independent(Normal(loc=Mut.unsqueeze(0),
                               scale=torch.sqrt(Sigma_t.unsqueeze(0))), 1).log_prob(X.unsqueeze(1))

        # Combine log operations and apply lambda values
        W = W + np.log(2 * np.pi) * self.n / 2  # improved numerics
        W = torch.clip(W, -50, 50)  # numerical stability
        W = torch.exp(W) * self.Lambda_values.unsqueeze(0)  # (B, num_nonzero)

        # Weighted sum - use broadcasting instead of expand
        W_unsqueezed = W.unsqueeze(-1)  # (B, num_nonzero, 1)
        u = (U * W_unsqueezed).sum(dim=1)  # (B, n)
        normalization = W.sum(dim=1, keepdim=True)  # (B, 1)

        return u / normalization

    def f(self, t, y):
        return self.calc_u(y, t)

    def g(self, t, y):
        return self.epsilon * torch.eye(self.n, device=self.device).repeat(y.shape[0], 1, 1)

    def calc_JOT(self):
        I = torch.ones(self.N0, self.N1, self.n, device=self.device)
        S0_exp = self.S0.unsqueeze(dim=1)
        S1_exp = self.S1.unsqueeze(dim=0)

        if self.epsilon > 0.:
            M_eps = I + torch.sqrt(
                I + 16 / (4 * self.epsilon ** 4) * S0_exp.expand(self.N0, self.N1, self.n) * S1_exp.expand(self.N0,
                                                                                                           self.N1,
                                                                                                           self.n))

            self.C_full = (ot.dist(self.Mu0, self.Mu1)  # We need all the terms to get the true OT cost.
                           + (S0_exp.expand(self.N0, self.N1, self.n) + S1_exp.expand(self.N0, self.N1, self.n)).sum(
                        dim=-1)
                           - self.epsilon ** 2 * (M_eps.sum(dim=-1) - M_eps.log().sum(dim=-1)
                                                  + S0_exp.expand(self.N0, self.N1, self.n).log().sum(dim=-1)
                                                  + S1_exp.expand(self.N0, self.N1, self.n).log().sum(dim=-1)))
        else:
            self.C_full = (ot.dist(self.Mu0, self.Mu1)  # GSB cost reduces to BW distance.
                           + torch.sqrt(S0_exp.expand(self.N0, self.N1, -1) * S1_exp.expand(self.N0, self.N1, -1)).sum(
                        dim=-1)
                           + S0_exp.expand(self.N0, self.N1, -1).sum(dim=-1)
                           + S1_exp.expand(self.N0, self.N1, -1).sum(dim=-1))

        # Use sparse representation for the sum
        i_idx = self.Lambda_indices_i
        j_idx = self.Lambda_indices_j
        return (self.Lambda_values * self.C_full[i_idx, j_idx]).sum()

    def calc_rho(self, t):
        # Calculate the state distribution at time t without integration
        # Reconstruct full Lambda for mixture
        Lambda_full = torch.zeros(self.N0, self.N1, device=self.device)
        Lambda_full[self.Lambda_indices_i, self.Lambda_indices_j] = self.Lambda_values

        mix = Categorical(Lambda_full.reshape(-1))
        rho = MixtureSameFamily(mix, Independent(Normal(loc=self.Mu1 * t + (1 - t) * self.Mu0,
                                                        scale=self.Sigma(t)[0]), 1))
        return rho

    def sample_rho(self, t, B):
        # Create B samples from the distribution of the state
        # at time t without simulating the dynamics.
        return self.calc_rho(t).sample([B])

    def calc_Jtrue(self, B=5000):
        # calculate the true transport cost
        J = torch.tensor(0., device=self.device)
        T = torch.linspace(0, 0.99, 100, device=self.device)
        for t in T:
            J += 0.01 * torch.mean(torch.linalg.vector_norm(self.calc_u(self.sample_rho(t, B), t), dim=1) ** 2, dim=0)

        return J