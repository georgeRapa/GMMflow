from src import *
import torch
import torchsde
from sklearn.mixture import GaussianMixture

from eot_benchmark.gaussian_mixture_benchmark import (
    get_guassian_mixture_benchmark_sampler,
    get_guassian_mixture_benchmark_ground_truth_sampler,
    get_test_input_samples,
)
from eot_benchmark.metrics import (
    compute_BW_UVP_by_gt_samples, compute_BW_by_gt_samples, calculate_cond_bw
)

import time

# DIM_B = [2, 16, 64, 128]
# EPS_B =  [0.1]
EPS = 0.1
DIM_B = [2, 16, 64, 128]
Ncomp = [5, 10, 20, 50, 100]


SEED = 987
BATCH_SIZE=1024
NUM_SAMPLES_PLOT=100
SELECTED_IDX = [233,43,12,62,555]
SELECTED_ITERS = 16
NUM_SAMPLES_METRICS1=50
NUM_SAMPLES_METRICS2=500

# torch.manual_seed(0), np.random.seed(0)

device = 'cuda' # use cpu for 128 dim

cBW_UVP = torch.zeros((len(DIM_B), len(Ncomp)))
BW_UVP = torch.zeros((len(DIM_B), len(Ncomp)))
run_times = torch.zeros((len(DIM_B), len(Ncomp)))
inference_times = torch.zeros((len(DIM_B), len(Ncomp)))


for i, DIM in enumerate(DIM_B):
    for j, ncomp in enumerate(Ncomp):

        torch.cuda.empty_cache()
        t0 = time.time()

        input_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="input", dim=DIM, eps=EPS,
                                                               batch_size=BATCH_SIZE, device=device, download=False)

        target_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="target", dim=DIM, eps=EPS,
                                                                batch_size=BATCH_SIZE, device=device, download=False)

        ground_truth_plan_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(dim=DIM, eps=EPS,
                                                                                        batch_size=BATCH_SIZE,
                                                                                        device=device,
                                                                                        download=False)
        input_samples = input_sampler.sample(10000)
        target_samples = target_sampler.sample(10000)
        print("GMMs ready")
        ### 2.1 Map points from P by Ground-Truth process

        gmm1 = GaussianMixture(n_components=ncomp, covariance_type='diag').fit(target_samples.cpu().detach().numpy())
        gmm0 = GaussianMixture(n_components=ncomp, covariance_type='diag').fit(input_samples.cpu().detach().numpy())

        Mu0 = torch.tensor(gmm0.means_, dtype=torch.float32, device=device)
        Sigma0 = torch.tensor(gmm0.covariances_, dtype=torch.float32, device=device)
        W0 = torch.tensor(gmm0.weights_, dtype=torch.float32, device=device)

        Mu1 = torch.tensor(gmm1.means_, dtype=torch.float32, device=device)
        Sigma1 = torch.tensor(gmm1.covariances_, dtype=torch.float32, device=device)
        W1 = torch.tensor(gmm1.weights_, dtype=torch.float32, device=device)

        # rho0 = MultivariateNormal(loc=Mu0[0], covariance_matrix=torch.diag(Sigma0[0]))
        # mix = Categorical(W1)
        # comp = Independent(Normal(Mu1, Sigma1), 1)
        # gmm1 = MixtureSameFamily(mix, comp)

        sde = GMMflow(Mu0, Mu1, Sigma0, Sigma1, W0=W0, W1=W1, Lambda=None, omega=np.sqrt(EPS), device=device)

        run_times[i,j] = time.time() - t0

        print("SDE ready")

        # # y0 = rho0.sample([500])
        y0 = input_sampler.sample(1).reshape(-1,DIM)
        # y0 = input_sampler.sample(NUM_SAMPLES_METRICS1)
        # Y0 = y0.unsqueeze(1).repeat(1, NUM_SAMPLES_METRICS2, 1) #.reshape(-1, DIM)
        #
        t = torch.linspace(0, 1., 2)
        # y = []

        t0 = time.time()
        yy = torchsde.sdeint(sde, y0, t, method='euler', dt=0.02)
        # for yy0 in tqdm(Y0):
        #     y.append(torchsde.sdeint(sde, yy0, t, method='euler'))

        inference_times[i,j] = time.time() - t0

        # y = torch.concat(y, dim=1)
        #
        print("GWTF solution ready")

        # Y1_gwtf = y[-1, :, :].reshape(NUM_SAMPLES_METRICS1, NUM_SAMPLES_METRICS2, DIM)
        # Y1_true = ground_truth_plan_sampler.conditional_plan.sample(Y0.reshape(-1, DIM)).reshape(NUM_SAMPLES_METRICS1,
        #                                                                         NUM_SAMPLES_METRICS2, DIM)
        # #
        # cBW_UVP_true = calculate_cond_bw(y0.float(), Y1_true.float(), eps=EPS, dim=DIM)
        # cBW_UVP[i, j] = calculate_cond_bw(y0.float(), Y1_gwtf.float(), eps=EPS, dim=DIM)

        # BW_UVP[i, j] = compute_BW_UVP_with_gt_stats(y0.float(), Y1_gwtf.float(), eps=EPS, dim=DIM)

        # print("comp NO = ", ncomp, "  D = ", DIM, "cBW_UVP = ", cBW_UVP[i, j], "(true is ", cBW_UVP_true, ") \n")


print(run_times, '\n', inference_times)


np.savez('run_times_anlysis.npy',
         run_times=run_times.cpu().numpy(),
         inference_times=inference_times.cpu().numpy())
         # cBW_UVP = cBW_UVP.cpu().numpy())

