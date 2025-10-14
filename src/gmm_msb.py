import torch
import numpy as np
import itertools
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import MixtureSameFamily
import mosek
from mosek.fusion import Matrix

class Gspline_SDP():
    def __init__(self, n, N, eps, nt=100, nbatch=0):

        self.Nt = (N - 1) * nt + 1
        self.N = N  # number of marginals
        self.n = n  # state dimension
        self.eps = eps
        self.T = np.linspace(0., self.N - 1, self.Nt)
        self.DT = np.diff(self.T)
        self.nbatch = nbatch

        A = np.array([[0., 1.], [0., 0.]])
        self.Ac = np.kron(A, np.eye(self.n))
        B = np.array([[0.], [1.]])
        self.Bc = np.kron(B, np.eye(self.n))
        self.Dc = self.Bc * np.sqrt(eps)

        self.A = np.eye(self.n * 2) + self.Ac * self.DT[0]
        self.B = self.Bc * self.DT[0]
        self.D = self.Bc * np.sqrt(self.DT[0]) * np.sqrt(eps)
        self.build_OCS()
        if self.nbatch > 0:
            self.models = [self.M.clone() for _ in range(self.nbatch)]

    def build_OCS(self):

        import mosek
        from mosek.fusion import Matrix, Model, Domain, ObjectiveSense, Expr

        if isinstance(self.A, np.ndarray):
            n = self.A.shape[0]
            m = self.B.shape[1]
            Am = [Matrix.dense(self.A) for _ in range(self.Nt)]
            Bm = [Matrix.dense(self.B) for _ in range(self.Nt)]
            Dm = [Matrix.dense(self.D @ self.D.T) for _ in range(self.Nt)]
        else:
            pass

        self.M = Model()

        self.S = [self.M.variable(Domain.inPSDCone(n)) for _ in range(self.Nt)]
        self.Y = [self.M.variable(Domain.inPSDCone(m)) for _ in range(self.Nt - 1)]
        self.U = [self.M.variable([m, n], Domain.unbounded()) for _ in range(self.Nt - 1)]

        self.mu = [self.M.variable([n], Domain.unbounded()) for _ in range(self.Nt)]
        self.v = [self.M.variable([m], Domain.unbounded()) for _ in range(self.Nt - 1)]
        self.v_slack = [self.M.variable(1, Domain.greaterThan(0)) for _ in range(self.Nt - 1)]

        self.Sigma_par = [self.M.parameter('S'+str(i), [self.n, self.n]) for i in range(self.N)]
        self.Mu_par = [self.M.parameter('Mu'+str(i), self.n) for i in range(self.N)]

        # self.Sigma_par = [self.M.parameter([self.n, self.n]) for _ in range(self.N)]
        # self.Mu_par = [self.M.parameter(self.n) for _ in range(self.N)]

        self.J = Expr.zeros(1)
        self.Jpar = self.M.variable('Cost', 1)
        # self.Jpar = self.M.variable(1)

        constraint_expr = Expr.sub(self.S[0], 1e-3 * np.eye(n))
        self.M.constraint(constraint_expr, Domain.inPSDCone(n))

        for k in range(self.Nt - 1):
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
            self.J = Expr.add(self.J, Expr.mul(self.v_slack[k], self.DT[k]))

            V_constr = Expr.vstack(self.v_slack[k], Expr.constTerm(0.5), self.v[k])
            self.M.constraint(V_constr, Domain.inRotatedQCone(m + 2))

        for i in range(self.N):
            k = int(np.floor(i * (self.Nt - 1) / (self.N - 1)))
            sub_S = self.S[k].slice([0, 0], [self.n, self.n])
            self.M.constraint(Expr.sub(sub_S, self.Sigma_par[i]), Domain.equalsTo(0.0))
            sub_mu = self.mu[k].slice([0], [self.n])
            self.M.constraint(Expr.sub(sub_mu, self.Mu_par[i]), Domain.equalsTo(0.0))

        self.M.constraint(Expr.sub(self.Jpar, self.J), Domain.equalsTo(0.))
        self.M.objective(ObjectiveSense.Minimize, self.J)

    def solve_Gspline(self, S, Mu):

        for i in range(self.N):
            self.Sigma_par[i].setValue(S[i])
            self.Mu_par[i].setValue(Mu[i])

        self.M.solve()

        Sol = {"cov": [x.level().reshape((2 * self.n, 2 * self.n)) for x in self.S],
               "mu": [x.level() for x in self.mu],
               "U": [x.level().reshape((self.n, 2 * self.n)) for x in self.U],
               "v": [x.level() for x in self.v],
               "K": [self.U[i].level().reshape((self.n, 2 * self.n)) @ np.linalg.inv(
                   self.S[i].level().reshape((2 * self.n, 2 * self.n))) for i in
                     range(self.Nt - 1)],
               "cost": self.Jpar.level()}

        return Sol

    def solve_Batch(self, S, Mu):

        from mosek.fusion import Model

        lenS = len(S)
        if len(S) < len(self.models):
            S += [S[-1]] * (len(self.models) - len(S))


        for model, s, mu in zip(self.models, S, Mu):
            for i in range(self.N):
                model.getParameter('S'+str(i)).setValue(s[i])
                model.getParameter('Mu'+str(i)).setValue(mu[i])
                model.setSolverParam("numThreads", 2)

        status = Model.solveBatch(False,         # No race
                                  -1.0,          # No time limit
                                  self.nbatch*2,
                                  self.models)        # Array of Models to solve


        return [model.getVariable('Cost').level().item() for model in self.models[0:lenS]]


class GMMmSB():
    def __init__(self, S, Mu, W=None, eps=0., nt=10, nt_fine=100, Lambda=None, nbatch=10):

        self.S = S
        self.Mu = Mu
        self.eps = eps
        self.nt = nt
        self.nt_fine = nt_fine
        self.nbatch = nbatch

        self.N = len(S)
        self.Nc = np.array([len(self.S[i]) for i in range(self.N)], dtype=np.int32)
        self.n = self.S[0][0].shape[0]

        self.noise_type = "general"
        self.sde_type = "ito"

        if W is None:
            self.W = [1 / nc for nc in self.Nc]
        else:
            self.W = W

        self.GS = Gspline_SDP(n=self.n, N=self.N, eps=eps, nt=nt, nbatch=self.nbatch)
        self.GS_fine = Gspline_SDP(n=self.n, N=self.N, eps=eps, nt=nt_fine)

        self.T = self.GS.T
        self.T_fine = self.GS_fine.T

        self.A = torch.tensor(self.GS.Ac, dtype=torch.float32)
        self.B = torch.tensor(self.GS.Bc, dtype=torch.float32)
        self.D = torch.tensor(self.GS.Dc, dtype=torch.float32)

        if Lambda is None:
            if self.nbatch > 0:
                self.compute_cost_batch()
            else:
                self.compute_cost()
            self.compute_lambda()
        else:
            self.Lambda = Lambda

        self.posIndx = self.Lambda.nonzero()
        self.compute_fine()

    def compute_fine(self):

        self.sols_fine = []

        for indx in tqdm(self.posIndx):
            Cov = [c[i] for c, i in zip(self.S, indx)]
            Mean = [m[i] for m, i in zip(self.Mu, indx)]

            sol = self.GS_fine.solve_Gspline(Cov, Mean)  # could make it faster by //.
            sol['indx'] = indx
            self.sols_fine.append(sol)

    def calc_u_fine(self, x, t):

        B = x.shape[0]

        if torch.is_tensor(t):
            k = int(np.floor(t.item() / self.GS_fine.DT[0]))
        else:
            k = int(np.floor(t / self.GS_fine.DT[0]))

        u = torch.zeros((B, self.n), dtype=torch.float32)
        S_w = torch.zeros(B, dtype=torch.float32)

        for sol in self.sols_fine:
            v = torch.tensor(sol["v"][k], dtype=torch.float32)
            K = torch.tensor(sol["K"][k], dtype=torch.float32)
            mu = torch.tensor(sol["mu"][k], dtype=torch.float32)
            Sigma = torch.tensor(sol["cov"][k], dtype=torch.float32)
            indx = tuple(sol["indx"])

            ui = (K @ (x - mu).T).T + v
            w = MultivariateNormal(loc=mu, covariance_matrix=Sigma).log_prob(x).clip(-50., 50.)
            w = self.Lambda[indx] * torch.exp(w)

            u += ui * w.unsqueeze(1)

            S_w += w

        return u / S_w.unsqueeze(1)

    def compute_cost(self):

        self.C = np.empty(self.Nc)

        self.sols = []

        self.Indx = list(itertools.product(*[list(range(i)) for i in self.Nc]))

        for indx in tqdm(self.Indx):
            Cov = [c[i] for c, i in zip(self.S, indx)]
            Mean = [m[i] for m, i in zip(self.Mu, indx)]

            sol = self.GS.solve_Gspline(Cov, Mean)  # could make it faster by //.
            sol['indx'] = indx
            self.sols.append(sol)
            self.C[indx] = sol['cost'][0]

    def compute_lambda(self):

        import mosek
        from mosek.fusion import Matrix, Model, Domain, ObjectiveSense, Expr

        self.OT = Model()
        self._Lambda = self.OT.variable(Domain.greaterThan(0., self.Nc))

        for i in range(self.N):
            axes_to_sum_over = np.array(range(self.N), dtype=np.int32)
            axes_to_sum_over = np.delete(axes_to_sum_over, i)

            marginal_expr = self._Lambda
            for axis in sorted(axes_to_sum_over, reverse=True):
                marginal_expr = Expr.sum(marginal_expr, axis)

            self.OT.constraint(Expr.sub(marginal_expr, self.W[i]), Domain.equalsTo(0.0))

        self._J = self.OT.variable(Domain.unbounded())
        dot_product_expr = Expr.dot(Expr.flatten(self._Lambda), self.C.flatten())
        self.OT.constraint(Expr.sub(self._J, dot_product_expr), Domain.equalsTo(0.0))
        self.OT.objective(ObjectiveSense.Minimize, self._J)
        self.OT.solve()

        self.Lambda = torch.tensor(self._Lambda.level().reshape(self.Nc), dtype=torch.float32)

        self.Lambda[self.Lambda < 0.] = torch.tensor(0.)

    def calc_u(self, x, t):

        B = x.shape[0]

        if torch.is_tensor(t):
            k = int(np.floor(t.item() / self.GS.DT[0]))
        else:
            k = int(np.floor(t / self.GS.DT[0]))

        u = torch.zeros((B, self.n), dtype=torch.float32)
        S_w = torch.zeros(B, dtype=torch.float32)

        for i, indx in enumerate(self.Indx):

            if self.Lambda[indx] > 0.:
                v = torch.tensor(self.sols[i]["v"][k], dtype=torch.float32)
                K = torch.tensor(self.sols[i]["K"][k], dtype=torch.float32)
                mu = torch.tensor(self.sols[i]["mu"][k], dtype=torch.float32)
                Sigma = torch.tensor(self.sols[i]["cov"][k], dtype=torch.float32)

                ui = (K @ (x - mu).T).T + v
                w = MultivariateNormal(loc=mu, covariance_matrix=Sigma).log_prob(x).clip(-50., 50.)
                w = self.Lambda[indx] * torch.exp(w)

                u += ui * w.unsqueeze(1)

                S_w += w

        return u / S_w.unsqueeze(1)

    def compute_cost_batch(self):

        from more_itertools import chunked

        C = []

        self.Indx = list(itertools.product(*[list(range(i)) for i in self.Nc]))

        for indxs in tqdm(list(chunked(self.Indx, self.nbatch))):

            Cov = [[c[i] for c, i in zip(self.S, indx)] for indx in indxs]
            Mean  = [[m[i] for m, i in zip(self.Mu, indx)] for indx in indxs]

            cost = self.GS.solve_Batch(Cov, Mean) # could make it faster by //.

            C.append(np.array(cost))

        self.c = np.concatenate(C)
        self.C = np.concatenate(C).reshape(self.Nc)


    def sample_rho(self, t, B):

        if torch.is_tensor(t):
            k = int(np.floor(t.item() / self.GS.DT[0]))
        else:
            k = int(np.floor(t / self.GS.DT[0]))

        Means = []
        Covs = []

        for sol in self.sols_fine:
            mu = torch.tensor(sol["mu"][k], dtype=torch.float32)
            Sigma = torch.tensor(sol["cov"][k], dtype=torch.float32)

            Means.append(mu)
            Covs.append(Sigma)

        Means = torch.stack(Means)
        Covs = torch.stack(Covs)
        mix = Categorical(self.Lambda[tuple(self.posIndx.T)].reshape(-1))
        rho = MixtureSameFamily(mix, MultivariateNormal(loc=Means, covariance_matrix=Covs))
        return rho.sample([B])

    def f(self, t, y):
        return (self.A @ y.T).T + (self.B @ self.calc_u_fine(y, t).T).T

    def g(self, t, y):
        return self.D.repeat(y.shape[0], 1, 1)

    def fill_vel(self, X, t):

        if torch.is_tensor(t):
            k = int(np.floor(t.item()/self.GS_fine.DT[0]))
        else:
            k = int(np.floor(t/self.GS_fine.DT[0]))

        V = []

        for x in X:

            MuX = []
            SX = []
            MuV = []
            SV = []

            for sol in self.sols_fine:

                mu = torch.tensor(sol["mu"][k], dtype=torch.float32)
                Sigma = torch.tensor(sol["cov"][k], dtype=torch.float32)

                S11 = Sigma[0:self.n, 0:self.n]
                S12 = Sigma[0:self.n, self.n:]
                S22 = Sigma[self.n:, self.n:]
                m1 = mu[0:self.n]
                m2 = mu[self.n:]

                MuX.append(m1)
                SX.append(S11)
                iS = torch.linalg.inv(S11)
                MuV.append(m2  + S12.T @ iS @ ( x - m1))
                sV = S22 - S12.T @ iS @ S12
                SV.append((sV + sV.T)/2)


            means = torch.stack(MuX)
            covs = torch.stack(SX)
            weights = MultivariateNormal(loc=means, covariance_matrix=covs).log_prob(x.unsqueeze(0).repeat(len(self.sols_fine), 1))
            weights = torch.exp(weights)

            meansV = torch.stack(MuV)
            covsV  = torch.stack(SV)

            mix = Categorical(self.Lambda[tuple(self.posIndx.T)].reshape(-1) * weights)
            rho = MixtureSameFamily(mix, MultivariateNormal(loc=meansV, covariance_matrix=covsV))

            v = rho.sample([1]).squeeze(0)

            V.append(v)

        V = torch.stack(V)

        return torch.concat([X, V], dim=1)


__all__ = ['Gspline_SDP', 'GMMmSB']
