import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import ot
import bisect
import numpy as np
import mosek
from mosek.fusion import *
# import mosek.fusion.pythonic # not necessary
from torch.distributions import Categorical
from torch.distributions import MixtureSameFamily



class OCS_SDP():
    """Base class for solving the GSB for a general LTI system using Semidefinite programming.
    Requires MOSEK with a valid license (see https://docs.mosek.com/11.0/pythonfusion/install-interface.html)."""

    def __init__(self, Mu0, Mu1, Sigma0, Sigma1, A, B, D, N=101):

        self.N0 = Mu0.shape[0]
        self.N1 = Mu1.shape[0]

        self.Mu0 = np.array(Mu0, dtype=np.float64)
        self.Mu1 = np.array(Mu1, dtype=np.float64)
        self.Sigma0 = np.array(Sigma0, dtype=np.float64)
        self.Sigma1 = np.array(Sigma1, dtype=np.float64)

        self.T = np.linspace(0., 1., N)
        self.DT = np.diff(self.T)
        self.N = N
        self.n = A.shape[0]
        self.m = B.shape[1]

        self.A = np.eye(self.n) + A * self.DT[0]
        self.B = B * self.DT[0]
        self.D = D * np.sqrt(self.DT[0])
        self.build_OCS()

    def build_OCS(self):

        """Builds a discrete-time Optimal Covariance Steering problem following
        https://arxiv.org/pdf/2302.14296"""

        n = self.n
        m = self.m
        Am = [Matrix.dense(self.A) for _ in range(self.N)] # Supports time-varying A, B, D if needed.
        Bm = [Matrix.dense(self.B) for _ in range(self.N)]
        Dm = [Matrix.dense(self.D @ self.D.T) for _ in range(self.N)]


        # %%
        self.M = Model()

        self.S = [self.M.variable(Domain.inPSDCone(n)) for _ in range(self.N)]
        self.Y = [self.M.variable(Domain.inPSDCone(m)) for _ in range(self.N - 1)]
        self.U = [self.M.variable([m, n], Domain.unbounded()) for _ in range(self.N - 1)]

        self.mu = [self.M.variable([n], Domain.unbounded()) for _ in range(self.N)]
        self.v = [self.M.variable([m], Domain.unbounded()) for _ in range(self.N - 1)]
        self.v_slack = [self.M.variable(1, Domain.greaterThan(0)) for _ in range(self.N - 1)]

        self.mui = self.M.parameter(n)
        self.muf = self.M.parameter(n)
        self.Si = self.M.parameter([n, n])
        self.Sf = self.M.parameter([n, n])
        # %%
        self.J = Expr.zeros(1)
        self.Jpar = self.M.variable(1)

        for k in range(self.N - 1):
            constr = Expr.neg(self.S[k + 1])
            constr = Expr.add(constr, Expr.mul(Expr.mul(Am[k], self.S[k]), Matrix.transpose(Am[k])))
            constr = Expr.add(constr, Expr.mul(Expr.mul(Bm[k], self.U[k]), Matrix.transpose(Am[k])))
            constr = Expr.add(constr, Expr.mul(Expr.mul(Am[k], Matrix.transpose(self.U[k])), Matrix.transpose(Bm[k])))
            constr = Expr.add(constr, Expr.mul(Expr.mul(Bm[k], self.Y[k]), Matrix.transpose(Bm[k])))
            constr = Expr.add(constr, Dm[k])
            self.M.constraint(constr, Domain.equalsTo(0.))

            X = Expr.stack([[self.S[k], Expr.transpose(self.U[k])], [self.U[k], self.Y[k]]])
            self.M.constraint(X, Domain.inPSDCone(n + m))

            constr = Expr.neg(self.mu[k + 1])
            constr = Expr.add(constr, Expr.mul(Am[k], self.mu[k]))
            constr = Expr.add(constr, Expr.mul(Bm[k], self.v[k]))
            self.M.constraint(constr, Domain.equalsTo(0.))

            self.J = Expr.add(self.J, Expr.mul(Expr.sum(self.Y[k].diag()), self.DT[k]))
            # J = Expr.add(J, Expr.sum(S[k].diag()))
            self.J = Expr.add(self.J, Expr.mul(self.v_slack[k], self.DT[k]))

            V_constr = Expr.stack(0, [self.v_slack[k], Expr.constTerm(0.5), self.v[k]])
            self.M.constraint(V_constr, Domain.inRotatedQCone(m + 2))

        # self.M.constraint(self.S[0], Domain.equalsTo(self.Si))
        # self.M.constraint(self.S[-1], Domain.equalsTo(self.Sf))

        self.M.constraint(Expr.sub(self.S[0], self.Si), Domain.equalsTo(0.))
        self.M.constraint(Expr.sub(self.S[-1], self.Sf), Domain.equalsTo(0.))

        self.M.constraint(Expr.sub(self.mu[0], self.mui), Domain.equalsTo(0.))
        self.M.constraint(Expr.sub(self.mu[-1], self.muf), Domain.equalsTo(0.))
        # %%
        # Objective
        self.M.constraint(Expr.sub(self.Jpar, self.J), Domain.equalsTo(0.))
        self.M.objective(ObjectiveSense.Minimize, self.J)
        # %%

    def calcTransportCost(self):
        self.transportCost = np.empty((self.N0, self.N1))

        self.Sol = [[None] * self.N1 for _ in range(self.N0)]
        i = 0
        for mui, Si in zip(self.Mu0, self.Sigma0):
            j = 0
            for muf, Sf in zip(self.Mu1, self.Sigma1):
                # print("solving problem ", i, ", ", j)
                self.mui.setValue(mui)
                self.muf.setValue(muf)
                self.Si.setValue(Si)
                self.Sf.setValue(Sf)

                self.M.solve()
                self.Sol[i][j] = {"cov": [x.level().reshape((self.n, self.n)) for x in self.S],
                                  "mu": [x.level() for x in self.mu],
                                  "U": [x.level().reshape((self.m, self.n)) for x in self.U],
                                  "v": [x.level() for x in self.v],
                                  "K": [self.U[i].level().reshape((self.m, self.n)) @ np.linalg.inv(
                                      self.S[i].level().reshape((self.n, self.n))) for i in
                                        range(self.N - 1)]}

                self.transportCost[i, j] = self.Jpar.level()
                j += 1
            i += 1


class GMMflow_SDP(nn.Module):
    """ GMMflow with general LTI prior dynamics and full covariances"""

    def __init__(self, Mu0, Mu1, Sigma0, Sigma1, W0, W1, A, B, D, N=101):
        super().__init__()

        self.n = A.shape[0]
        self.m = B.shape[1]
        self.A = torch.tensor(A, dtype=torch.float32)
        self.B = torch.tensor(B, dtype=torch.float32)
        self.D = torch.tensor(D, dtype=torch.float32)

        self.Mu0 = Mu0
        self.Mu1 = Mu1

        self.N0 = Mu0.shape[0]
        self.N1 = Mu1.shape[0]

        if W0 is None:
            self.W0 = torch.ones(self.N0)/self.N0
        else:
            self.W0 = W0

        if W1 is None:
            self.W1 = torch.ones(self.N1)/self.N1
        else:
            self.W1 = W1

        self.noise_type = "general"
        self.sde_type = "ito"

        self.ocs = OCS_SDP(Mu0, Mu1, Sigma0, Sigma1, A, B, D, N=N)
        self.T = self.ocs.T
        self.DT = self.ocs.DT[0]
        self.ocs.calcTransportCost()
        self.C = torch.tensor(self.ocs.transportCost)
        self.calc_coupling(self.C)

    def calc_coupling(self, C):

        self.Lambda = ot.emd(self.W0,
                             self.W1,
                             C)

    def calc_JOT(self):

        J = torch.tensor(0.)
        for i in range(self.N0):
            for j in range(self.N1):
                J += self.Lambda[i, j] * self.C[i,j]

        return J

    def sample_rho(self, t, B):

        k = bisect.bisect_left(self.T, t)
        M = torch.zeros(self.N0*self.N1, self.n)
        S = torch.zeros(self.N0*self.N1, self.n, self.n)

        count = 0
        for i in range(self.N0):
            for j in range(self.N1):
                mu = torch.tensor(self.ocs.Sol[i][j]["mu"][k], dtype=torch.float32).unsqueeze(dim=0)
                Sigma = torch.tensor(self.ocs.Sol[i][j]["cov"][k], dtype=torch.float32).unsqueeze(dim=0)

                M[count, :] = mu
                S[count, :, :] = Sigma
                count +=1

        mix = Categorical(self.Lambda.reshape(-1))
        rho = MixtureSameFamily(mix, MultivariateNormal(loc=M, covariance_matrix=S))
        return rho.sample([B])

    def calc_Jtrue(self, B=5000):

        J = torch.tensor(0.)
        for t in self.T[0:-1]:
            J += self.DT * torch.mean(torch.linalg.vector_norm(self.calc_u(self.sample_rho(t, B), t), dim=1)**2, dim=0)

        return J

    def calc_rho(self, x, t):

        k = bisect.bisect_left(self.T, t)

        rho = torch.zeros_like(x[:, 0])

        for i in range(self.N0):
            for j in range(self.N1):

                mu = torch.tensor(self.ocs.Sol[i][j]["mu"][k], dtype=torch.float32)
                Sigma = torch.tensor(self.ocs.Sol[i][j]["cov"][k], dtype=torch.float32)

                rho_i = torch.exp(MultivariateNormal(loc=mu,covariance_matrix=Sigma).log_prob(x))
                rho += rho_i * self.Lambda[i,j]

        return rho

    def calc_u(self, x, t):

        B = x.shape[0]
        k = min(bisect.bisect_left(self.T, t), len(self.T) - 2)
        u = torch.zeros((B, self.m), dtype=torch.float32)
        S_w = torch.zeros(B, dtype=torch.float32)

        for i in range(self.N0):
            for j in range(self.N1):
                v = torch.tensor(self.ocs.Sol[i][j]["v"][k], dtype=torch.float32)
                K = torch.tensor(self.ocs.Sol[i][j]["K"][k], dtype=torch.float32)
                mu = torch.tensor(self.ocs.Sol[i][j]["mu"][k], dtype=torch.float32)
                Sigma = torch.tensor(self.ocs.Sol[i][j]["cov"][k], dtype=torch.float32)

                ui = (K @ (x - mu).T).T + v
                w = self.Lambda[i, j] * torch.exp(MultivariateNormal(loc=mu,
                                                                     covariance_matrix=Sigma).log_prob(x))
                u += ui * w.unsqueeze(1)

                S_w += w

        return u / S_w.unsqueeze(1)

    def f(self, t, y):
        return (self.A @ y.T).T + (self.B @ self.calc_u(y, t).T).T

    def g(self, t, y):
        # return (self.D @ torch.ones_like(y).T).T # for diagonal noise
        return self.D.repeat(y.shape[0], 1, 1)