import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pcqi import compute_pcqi

# paths
output_dir = r"C:\ml\project\output"
gt_dir = r"C:\ml\project\data\testB"

def compute_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)

def compute_uciqe(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = img_lab[:, :, 0] / 255.0
    a = img_lab[:, :, 1] / 255.0
    b = img_lab[:, :, 2] / 255.0

    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    con_l = np.max(L) - np.min(L)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    mu_s = np.mean(s)

    # standard UCIQE formula coefficients
    uciqe = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
    return float(uciqe)

def eme(channel, block_size=8):
    h, w = channel.shape
    h_crop = h - (h % block_size)
    w_crop = w - (w % block_size)
    channel = channel[:h_crop, :w_crop]

    blocks = []
    for i in range(0, h_crop, block_size):
        for j in range(0, w_crop, block_size):
            block = channel[i:i+block_size, j:j+block_size]
            Imax = np.max(block)
            Imin = np.min(block)
            if Imin == 0:
                Imin = 1e-6
            if Imax > 0:
                blocks.append(np.log(Imax / Imin))
    if len(blocks) == 0:
        return 0.0
    return 2.0 * np.mean(blocks)

def compute_uiqm(img):
    # img expected RGB uint8
    img = img.astype(np.float32)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    rg = R - G
    yb = (R + G) / 2 - B

    mu_rg = np.mean(rg)
    mu_yb = np.mean(yb)
    sigma_rg = np.std(rg)
    sigma_yb = np.std(yb)

    uicm = -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(sigma_rg**2 + sigma_yb**2)

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    uism = eme(gradient_mag, block_size=8)

    R_eme = eme(R, block_size=8)
    G_eme = eme(G, block_size=8)
    B_eme = eme(B, block_size=8)
    uiconm = (R_eme + G_eme + B_eme) / 3.0

    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return float(uiqm)

results = []

for img_name in os.listdir(output_dir):
    out_path = os.path.join(output_dir, img_name)
    gt_path = os.path.join(gt_dir, img_name)

    if not os.path.exists(gt_path):
        print(f"Missing GT for {img_name}")
        continue

    out_img = cv2.imread(out_path)
    gt_img = cv2.imread(gt_path)

    if out_img is None or gt_img is None:
        print(f"Could not read {img_name}")
        continue

    if out_img.shape != gt_img.shape:
        gt_img = cv2.resize(gt_img, (out_img.shape[1], out_img.shape[0]))

    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    psnr = peak_signal_noise_ratio(gt_img, out_img, data_range=255)
    ssim = structural_similarity(gt_img, out_img, channel_axis=2, data_range=255)
    ie = compute_entropy(out_img)
    uiqm = compute_uiqm(out_img)
    uciqe = compute_uciqe(out_img)
    pcqi = compute_pcqi(gt_img, out_img)

    results.append([img_name, psnr, ssim, ie, uiqm, uciqe, pcqi])

df = pd.DataFrame(results, columns=["Image", "PSNR", "SSIM", "IE", "UIQM", "UCIQE", "PCQI"])

save_path = r"C:\ml\project\evaluation\results\metrics_all_parameters.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)

print("\nAverage Results:")
print(df.mean(numeric_only=True))