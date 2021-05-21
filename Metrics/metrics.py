from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error,peak_signal_noise_ratio


import cv2
import os

os.chdir(os.getcwd())
#original = cv2.imread("test_p7_.jpg")
#classical = cv2.imread("test_p7_restored.jpg")
#cgan = cv2.imread("c3de432bc9f1537b706b0d5d.png")

original = cv2.imread("test_p8_.jpg")
classical = cv2.imread("test_p8_restored.jpg")
cgan = cv2.imread("511791af981c37ea5f6810c5.png")



mse_classical = mean_squared_error(original, classical)
mse_cgan = mean_squared_error(original, cgan)

ssim_classical = ssim(original, classical,  data_range=classical.max() - classical.min(),multichannel=True)
ssim_cgan = ssim(original, cgan,  data_range=cgan.max() - cgan.min(),multichannel=True)


psnr_classical=  peak_signal_noise_ratio(original, classical,  data_range=classical.max() - classical.min())
psnr_cgan =  peak_signal_noise_ratio(original, cgan,  data_range=cgan.max() - cgan.min())


print("Structural similarity Index for Classical Approach",ssim_classical*100)
print("Structural similarity Index for Unsupervised Approach",ssim_cgan*100)


print("Mean square error for Classical Approach",mse_classical*100)
print("Mean square error for Unsupervised Approach",mse_cgan*100)


print("Peak Signal to Noise ratio for classical approach",psnr_classical)
print("Peak Signal to Noise ratio for Unsupervised approach",psnr_cgan)
