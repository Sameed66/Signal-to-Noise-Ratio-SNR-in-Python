import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load a color image
image = cv2.imread('face.png')  # Replace 'example.jpg' with your image file
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV default) to RGB

# Resize the image to 256x256 if necessary
if image.shape[0] != 256 or image.shape[1] != 256:
    image = cv2.resize(image, (256, 256))

# Display the type of the data
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)

# Display the image
plt.figure(figsize=(4, 4))
plt.imshow(image)
plt.title("Original Color Image")
plt.axis('off')
plt.show()


# Calculate and plot histograms for each channel
colors = ['Red', 'Green', 'Blue']
plt.figure(figsize=(10, 4))
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, label=f'{color} Channel', color=color.lower())
plt.title("Histograms of R, G, B Channels")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Convert to grayscale using the formula (R/3 + G/3 + B/3)
gray = (image[:, :, 0] / 3 + image[:, :, 1] / 3 + image[:, :, 2] / 3).astype(np.double)

# Plot the grayscale image
plt.figure(figsize=(6, 6))
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image (R/3 + G/3 + B/3)")
plt.axis('off')
plt.show()

# Histogram of grayscale image
plt.hist(gray.ravel(), bins=256,  color='gray')
plt.title("Histogram of Grayscale Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

# Extract minimum and maximum grayscale values
min_gray = np.min(gray)
max_gray = np.max(gray)
print(f"Minimum Grayscale Value: {min_gray}")
print(f"Maximum Grayscale Value: {max_gray}")

# Define noise parameters
mean = 0
std_dev = 25  # Standard deviation

# Generate Gaussian noise
noise = np.random.normal(mean, std_dev, gray.shape).astype(np.float32)

# Display the noise image
plt.imshow(noise, cmap='gray')
plt.title("Gaussian White Noise")
plt.axis('off')
plt.show()

# Histogram of noise
plt.hist(noise.ravel(), bins=256, color='gray', alpha=0.7)
plt.title("Histogram of Noise Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

# Compute mean and standard deviation
mean_noise = np.mean(noise)
std_dev_noise = np.std(noise)
print(f"Mean of Noise: {mean_noise:.2f}")
print(f"Standard Deviation of Noise: {std_dev_noise:.2f}")

from scipy.signal import correlate2d

# Compute autocorrelation
autocorr = correlate2d(noise, noise, mode='full')

# Display the autocorrelation
plt.imshow(autocorr, cmap='hot')
plt.title("Autocorrelation of Noise")
plt.colorbar()
plt.axis('off')
plt.show()

# Add noise to grayscale image (wrong)
noisy_image = gray + noise

# Display histogram of the noisy image
plt.hist(noisy_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.title("Histogram of Noisy Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

# Compute SNR and PSNR
snr = (max_gray - min_gray) / std_dev_noise
psnr_db = 20 * np.log10(snr)

print(f"SNR: {snr:.2f}")
print(f"PSNR (in dB): {psnr_db:.2f}")

from scipy.ndimage import convolve

# Define a normalized 2D filter
filter_size = 3
h = np.ones((filter_size, filter_size)) / (filter_size ** 2)

# Apply filter
smoothed_noisy = convolve(noisy_image, h)
smoothed_original = convolve(gray, h)

# Compute new SNR
std_smoothed_noise = np.std(smoothed_noisy - smoothed_original)
new_snr = (max_gray - min_gray) / std_smoothed_noise
print(f"New SNR after smoothing: {new_snr:.2f}")

# Varying filter sizes
filter_sizes = [3, 5, 7, 9]
snr_values = []

for size in filter_sizes:
    h = np.ones((size, size)) / (size ** 2)
    smoothed_noisy = convolve(noisy_image, h)
    smoothed_original = convolve(gray, h)
    std_smoothed_noise = np.std(smoothed_noisy - smoothed_original)
    snr = (max_gray - min_gray) / std_smoothed_noise
    snr_values.append(snr)

# Plot SNR evolution
plt.plot(filter_sizes, snr_values, marker='o')
plt.title("SNR Evolution with Filter Size")
plt.xlabel("Filter Size")
plt.ylabel("SNR")
plt.grid()
plt.show()












