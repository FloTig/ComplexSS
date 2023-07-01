import numpy as np
import matplotlib.pyplot as plt


def find_freqs(array, N_largest_freqs=20):
    # Plot the 2D fourier transform of the final concentration distribution
    fft = np.abs(np.real(np.fft.rfft2(array)))
    fft[0,0] = 0  # Remove the measurement of the average (freq 0 component)

    # Find largest frequency components
    l_freq_idx = np.argsort(fft, axis=None)
    x_freqs = np.unravel_index(l_freq_idx, fft.shape)[1]
    y_freqs = np.unravel_index(l_freq_idx, fft.shape)[0]
    # x_freqs = [i % fft.shape[0] for i in l_freq_idx]
    # y_freqs = [i // fft.shape[1] for i in l_freq_idx]

    # Only keep the most prevalent ones that describe positive frequencies
    largest_x_freqs = np.array([f for f in x_freqs if f < fft.shape[1] // 2][-N_largest_freqs:])
    largest_y_freqs = np.array([f for f in y_freqs if f < fft.shape[0] // 2][-N_largest_freqs:])
    largest_amplitudes = fft[largest_y_freqs, largest_x_freqs]

    return largest_x_freqs, largest_y_freqs, largest_amplitudes


def freq_to_lengths(shape, x_freqs, y_freqs):
    # Turn the frequency into wavelengths (in pixels)
    # Avoid dividing by zero, cap large wavelengths to the gridsize
    x_lengths = np.clip(np.divide(shape[1], x_freqs + 1e-20), 0, shape[0]*2)
    y_lengths = np.clip(np.divide(shape[0], y_freqs + 1e-20), 0, shape[1]*2)

    # Calculate norm of the "wave vector"
    wavelengths = np.sqrt(np.power(x_lengths, 2) + np.power(y_lengths, 2))

    return x_lengths, y_lengths, wavelengths


def plot_freqs(array, ax, N_largest_freqs=20):
    x_freqs, y_freqs, amplitudes = find_freqs(array, N_largest_freqs)

    ax.set_title("Frequencies")
    fft_scatter = ax.scatter(x_freqs, y_freqs, c=amplitudes, cmap=plt.cm.winter)
    return fft_scatter


def plot_lengths(array, ax, N_largest_freqs=20):
    x_freqs, y_freqs, amplitudes = find_freqs(array, N_largest_freqs)
    x_lengths, y_lengths, wavelengths = freq_to_lengths(array.shape, x_freqs, y_freqs)

    ax.set_title("Wavelengths")
    ax.scatter(x_lengths, amplitudes, c=amplitudes, cmap=plt.cm.winter)
    ax.scatter(y_lengths, amplitudes, c=amplitudes, cmap=plt.cm.winter)


if __name__ == "__main__":
    f = np.zeros((20, 30))
    f[0,3] = 1
    f[1,0] = 1
    w = np.real(np.fft.ifft2(f))

    fig, [imax, ax1, ax2] = plt.subplots(1,3)
    imax.imshow(np.real(w), cmap=plt.cm.winter)
    plot_freqs(w, ax1)
    plot_lengths(w, ax2)
    plt.show()


