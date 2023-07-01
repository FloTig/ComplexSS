import numpy as np
import matplotlib.pyplot as plt

# Generate data
freq = 7
xs = np.arange(0, 2*np.pi, 0.01)

noise = .8
sinus = np.sin(xs) + \
    np.sin(xs * (freq//2)) * 3 + \
    np.sin(xs * freq) * 2 + \
    noise * np.random.rand(xs.size)

# Estimate characteristic length (half wavelength) in pixels
fft = np.power(np.real(np.fft.rfft(sinus)[1:]), 2)
est_freq = np.argmax(fft)
est_len = xs.size / est_freq
print("Freq:", est_freq, "length:", est_len)

# Plot data and transform
fig, [ax1, ax2] = plt.subplots(1, 2)
ax1.plot(sinus)
ax2.scatter(range(fft.size), fft)

# Plot the expected max frequency
ylim = ax2.get_ylim()
ax2.vlines(freq, *ylim, color='red')

# Plot estimated wavelength
ylim = ax1.get_ylim()
ax1.vlines([i*est_len for i in range(freq)], *ylim, color='red')

plt.show()


# Understanding the fourier whatever the other way around
fourier_space = np.zeros((50,50))
fourier_space[5,1] = 1
fourier_space[5,0] = 1
fourier_space[0,1] = 1

realspace = np.fft.ifft2(fourier_space)
plt.imshow(np.real(realspace), cmap=plt.cm.winter)
plt.show()

# plt.imshow(np.imag(realspace), cmap=plt.cm.winter)
# plt.show()
