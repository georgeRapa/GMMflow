import gc

from old.NIPS.gwtf_source import *
import torchsde
from sklearn.mixture import GaussianMixture

from tqdm import tqdm
from eot_benchmark.gaussian_mixture_benchmark import (
    get_guassian_mixture_benchmark_sampler,
    get_guassian_mixture_benchmark_ground_truth_sampler,
    get_test_input_samples,
)
from eot_benchmark.metrics import (
    compute_BW_UVP_by_gt_samples, compute_BW_by_gt_samples, calculate_cond_bw
)

# %%
# DIM_B = [2, 16, 64, 128]
# EPS_B = [0.1, 1, 10]
DIM_B = [16]
EPS_B = [0.1]
SEED = 1 # 987
BATCH_SIZE = 1024
NUM_SAMPLES_PLOT = 1000
SELECTED_IDX = [233, 43, 12, 62, 555]
SELECTED_ITERS = 16
NUM_SAMPLES_METRICS1 = 10
mult = 100
NUM_SAMPLES_METRICS2 = 1000

torch.manual_seed(1024), np.random.seed(1024)

device = 'cuda'  # use cpu for 128 dim

cBW_UVP = torch.zeros((len(DIM_B), len(EPS_B)))
BW_UVP = torch.zeros((len(DIM_B), len(EPS_B)))

for i, DIM in enumerate(DIM_B):
    for j, EPS in enumerate(EPS_B):
        gc.collect()
        torch.cuda.empty_cache()
        input_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="input", dim=DIM, eps=EPS,
                                                               batch_size=BATCH_SIZE, device='cpu', download=False)

        target_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="target", dim=DIM, eps=EPS,
                                                                batch_size=BATCH_SIZE, device='cpu', download=False)

        ground_truth_plan_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(dim=DIM, eps=EPS,
                                                                                        batch_size=BATCH_SIZE,
                                                                                        device='cpu',
                                                                                        download=False)
        input_samples = input_sampler.sample(5000)
        target_samples = target_sampler.sample(5000)
        # print("GMMs ready")
        ### 2.1 Map points from P by Ground-Truth process

        gmm1 = GaussianMixture(n_components=5, covariance_type='diag').fit(target_samples.cpu().detach().numpy())
        gmm0 = GaussianMixture(n_components=1, covariance_type='diag').fit(input_samples.cpu().detach().numpy())

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

        # sde = GMMflow_analytic(Mu0, Mu1, Sigma0, Sigma1, W0=W0, W1=W1, cov_type='diag', Lambda=None, omega=np.sqrt(EPS),
        #                        device=device).to(device)
        sde = GMMflow_OP(Mu0, Mu1, Sigma0, Sigma1, W0=W0, W1=W1, Lambda=None, omega=np.sqrt(EPS), device=device)
        print("SDE ready")

        # y0 = rho0.sample([500])
        y0b = []
        Y0b = []
        Y1_gwtfb = []
        t = torch.linspace(0, 1., 2)

        for _ in tqdm(range(mult)):
            with torch.no_grad():

                gc.collect()
                torch.cuda.empty_cache()

                y0 = input_sampler.sample(NUM_SAMPLES_METRICS1).to(device)
                Y0 = y0.unsqueeze(1).repeat(1, NUM_SAMPLES_METRICS2, 1).reshape(-1, DIM)

                y = torchsde.sdeint(sde, Y0, t, method='euler')

                Y1_gwtf = y[-1, :, :].reshape(NUM_SAMPLES_METRICS1, NUM_SAMPLES_METRICS2, DIM)

                Y0b.append(Y0.detach().cpu())
                y0b.append(y0.detach().cpu())
                Y1_gwtfb.append(Y1_gwtf.detach().cpu())


        y0 = torch.concat(y0b, dim=0)
        Y0 = torch.concat(Y0b, dim=0)
        Y1_gwtf = torch.concat(Y1_gwtfb, dim=0)

        Y1_true = ground_truth_plan_sampler.conditional_plan.sample(Y0).reshape(mult*NUM_SAMPLES_METRICS1,
                                                                                NUM_SAMPLES_METRICS2, DIM)

        cBW_UVP_true = calculate_cond_bw(y0.float(), Y1_true.float(), eps=EPS, dim=DIM)
        cBW_UVP[i, j] = calculate_cond_bw(y0.float(), Y1_gwtf.float(), eps=EPS, dim=DIM)

        # print("EPS = ", EPS, "  D = ", DIM, "cBW_UVP = ", cBW_UVP[i, j], "(true is ", cBW_UVP_true, ") \n")

        BW_UVP[i, j] = compute_BW_UVP_by_gt_samples(y[-1].cpu().numpy(),
                                                    ground_truth_plan_sampler.conditional_plan.sample(Y0).cpu().numpy())
        print("EPS = ", EPS, "  D = ", DIM, "cBW_UVP = ", cBW_UVP[i, j], "(true is ", cBW_UVP_true, ")  ", "BW_UVP = ",
              BW_UVP[i, j], " \n")