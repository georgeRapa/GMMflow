from sklearn.mixture import GaussianMixture
from src import GMMflow
import torchsde
import time
from src.my_utils import *
from tqdm import tqdm
from PIL import Image
sys.path.append("third_party/LightSB")
sys.path.append("third_party/LightSB/ALAE")
from alae_ffhq_inference import load_model, encode, decode
import tracker
import os, shutil
from pytorch_fid import fid_score


# ###########
# you might need this for newer pytorch version > 2.6
# import argparse, tracker
# torch.serialization.add_safe_globals([tracker.RunningMeanTorch])
# #############

initial = "MAN" # MAN, WOMAN, ADULT, CHILDREN
translated = "WOMAN" # MAN, WOMAN, ADULT, CHILDREN

seed = 13
epsilon = 0.01
N0 = 10 # initial GMM components
N1 = 10 # final GMM components

X_train, X_test, Y_train, Y_test = load_latents(initial, translated, train_test_ratio=0.6)  # we need 10K test data to compute accurate FID.

print(f"Dataset loaded: {len(X_train)} X-training samples, {len(X_test)} X-test samples, {len(Y_train)} Y-training samples, {len(Y_test)} Y-test samples")
start = time.time()
gmm0 = GaussianMixture(n_components=N0,  covariance_type='diag', max_iter=200, random_state=0).fit(X_train)
gmm1 = GaussianMixture(n_components=N1,  covariance_type='diag', max_iter=200, random_state=0).fit(Y_train)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print('Using GPU for accelerated inference')
Mu0 = torch.tensor(gmm0.means_, dtype = torch.float32, device=device)
Sigma0 = torch.tensor(gmm0.covariances_, dtype = torch.float32, device=device)
Mu1 = torch.tensor(gmm1.means_, dtype = torch.float32, device=device)
Sigma1 = torch.tensor(gmm1.covariances_, dtype = torch.float32, device=device)

W0 = torch.tensor(gmm0.weights_, dtype = torch.float32, device=device)
W1 = torch.tensor(gmm1.weights_, dtype = torch.float32, device=device)

sde = GMMflow(Mu0, Mu1, Sigma0, Sigma1, W0, W1, omega=np.sqrt(epsilon), device = device)
end = time.time()
print("Training complete. Training time: ", end-start)

torch.cuda.empty_cache()
counter = 0

if device=='cuda':
    torch.cuda.empty_cache()

y0 = torch.tensor(X_test, dtype=torch.float32, device=device)[0:10000, :]

t = torch.linspace(0, 1., 2, device=device)

print("Performing batch inference:")
with torch.no_grad():
    yf = []
    for y0_batch in tqdm(torch.split(y0, 1000, dim=0)):
        if device=='cuda':
            torch.cuda.empty_cache()
        y = torchsde.sdeint(sde, y0_batch, t, method='euler', dt=0.01)  # y will be of size (t_size, batch_size, state_size)
        yf.append(y[-1, :, :].cpu())

yf = torch.cat(yf, dim=0).to(device)

model = load_model("third_party/LightSB/ALAE/configs/ffhq.yaml", training_artifacts_dir="third_party/LightSB/ALAE/training_artifacts/ffhq/")
model = model.to(device)

print("Batch decoding translated samples")

dir_path = "Images/GMMflow_W_10K/"
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)
    os.makedirs(dir_path)
else:
    os.makedirs(dir_path)


with torch.no_grad():
    decoded_img = []
    for mapped_batch in tqdm(torch.split(yf, 100, dim=0)):
        if device=='cuda':
            torch.cuda.empty_cache()
        dec = decode(model, mapped_batch).cpu()
        dec = ((dec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3,
                                                                                                       1).numpy()

        for i, im_arr in enumerate(dec):
            im = Image.fromarray(im_arr)
            im.save("Images/GMMflow_W_10K/" + str(counter) + "_" + str(i) + ".jpeg")

        counter += 1


print("Batch decoding true target samples")

dir_path = "Images/W_10K/"
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)
    os.makedirs(dir_path)
else:
    os.makedirs(dir_path)

ytrue = torch.tensor(Y_test, dtype=torch.float32, device=device)[0:10000, :]
with torch.no_grad():
    decoded_img = []
    for mapped_batch in tqdm(torch.split(ytrue, 100, dim=0)):
        torch.cuda.empty_cache()
        dec = decode(model, mapped_batch).cpu()
        dec = ((dec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3,
                                                                                                       1).numpy()

        for i, im_arr in enumerate(dec):
            im = Image.fromarray(im_arr)
            im.save("Images/W_10K/" + str(counter) + "_" + str(i) + ".jpeg")

        counter += 1

# to compute FID run python -m pytorch_fid Images/GMMflow_W_10K Images/W_10K --device cuda:0  or  use

FID = fid_score.calculate_fid_given_paths(paths=['Images/GMMflow_W_10K','Images/W_10K'],
                                          batch_size=128, # use the best you have
                                          device='cuda',
                                          dims=2048)  #dimensionality of Inception net latent layer

alae_bw = ALAE_BW(yf, ytrue)

print(f"FID is {FID}, ALAE_BW is {alae_bw}")