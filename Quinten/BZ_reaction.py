import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib import animation
from multiscale_complexity import calc_complexity

# Width, height of the image.
nx, ny = 256, 256
basis_rate = 1

ANIMATED = True
Nsteps = 100


def update(i, arr, c_lists, i_list, cmplx_list):
    """Updates the state of the BZ reaction after one timestep"""

    # Count the average amount of each species in the 9 cells around each cell
    # by convolution with the 3x3 array m.
    p = i % 2
    q = (p+1) % 2
    s = np.zeros((3, ny,nx))
    neighbor_kernel = np.ones((3,3)) / 9
    for k in range(3):
        s[k] = convolve2d(arr[p,k], neighbor_kernel, mode='same', boundary='wrap')

    # Append concentrations
    for k, c_list in enumerate(c_lists):
        c_list.append(arr[p,k].sum() / (nx * ny))

    # Calculate expensive metrics
    # if i % 50 == 0:
        # i_list.append(moran_i(arr[p,0]))
        # cmplx_list.append(calc_complexity(arr[p,0]))

    # Apply the reaction equations
    arr[q,0] = s[0] + s[0]*(alpha*s[1] - gamma*s[2])
    arr[q,1] = s[1] + s[1]*(beta*s[2] - alpha*s[0])
    arr[q,2] = s[2] + s[2]*(gamma*s[0] - beta*s[1])

    # Ensure the species concentrations are kept within [0,1].
    np.clip(arr[q], 0, 1, arr[q])
    return arr


def moran_i(grid):
    n = grid.size
    mean = np.mean(grid)
    deviations = grid - mean

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            deviations_matrix = deviations * deviations[i,j]

            # Calculate the distance matrix between grid cells
            distances = np.argwhere(np.ones_like(grid)) - np.array([i,j])
            distances[distances == 0] = 1e-10

    # Calculate the spatial lag matrix
    spatial_lag_matrix = deviations_matrix * (1 / distances)

    # Calculate Moran's I value
    moran_I = (n / (n - 1)) * (np.sum(spatial_lag_matrix) / np.sum(deviations_matrix))

    return moran_I


# Loop over different combinations of reaction rates.
for factor in [1]:#[0.5, 0.75, 1, 1.25, 1.5]:
    alpha, beta, gamma = basis_rate, basis_rate, basis_rate * factor

    # Initialize the array with random amounts of A, B and C.
    arr = np.random.random(size=(2, 3, ny, nx))

    # Initialize lists for saving concentration data
    concentrations = [[], [], []]
    moran_is = []
    complexities = []

    if ANIMATED:
        # Set up the image
        fig, (ax1, ax2) = plt.subplots(2,1)
        im = ax1.imshow(arr[0,0], cmap=plt.cm.winter)
        ax1.axis('off')

        line1, = ax2.plot([0], [0])
        line2, = ax2.plot([0], [0])
        line3, = ax2.plot([0], [0])

        def animate(i, arr, c_lists, i_list, cmplx_list):
            """Update the image for iteration i of the Matplotlib animation."""

            arr = update(i, arr, c_lists, i_list, complexities)
            im.set_array(arr[i % 2, 0])

            ax2.set_xlim(0, len(c_lists[0]))
            ax2.set_ylim(min(min(c_list) for c_list in c_lists) * 0.9, 1.1 * max(max(c_list) for c_list in c_lists))
            ax2.figure.canvas.draw()

            x = list(range(len(c_lists[0])))
            line1.set_data(x, c_lists[0])
            line2.set_data(x, c_lists[1])
            line3.set_data(x, c_lists[2])

            return [im, line1, line2, line3]


        anim = animation.FuncAnimation(fig, animate, frames=Nsteps, interval=1, blit=True, fargs=(arr, concentrations, moran_is, complexities))

        # To view the animation, uncomment this line
        plt.show()

        # To save the animation as an MP4 movie, uncomment this line
        # anim.save(filename='bz.mp4', fps=30)


    # Evolve the system without drawing anything
    else:
        for i in range(Nsteps):
            arr = update(i, arr, concentrations, moran_is, complexities)

    # Show plot of all concentrations over time
    plt.title("Total Concentrations")
    for i, c_list in enumerate(concentrations):
        plt.plot(c_list, label=str(i))
    plt.show()

    # Plot the final concentration distribution
    plt.title("Concentration of substance 1")
    im = plt.imshow(arr[0,0], cmap=plt.cm.winter)
    plt.show()

    # 3D attractor plot
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(*concentrations)
    # plt.show()

    # Print (relative) simulation parameters and final concentrations
    print(alpha, beta, gamma)
    print([f'{c / max(alpha, beta, gamma):.2f}' for c in [alpha, beta, gamma]])
    final_cs = [c[-1] for c in concentrations]
    print(final_cs)
    print([f'{c / max(final_cs):.2f}' for c in final_cs])

    # Plot the Moran I measure over time
    # plt.plot(moran_is)
    # plt.show()

    # plt.title("Multiscale Complexity")
    # plt.plot(complexities)
    # plt.show()

    # 2D FOURIER
    # Plot the 2D fourier transform of the final concentration distribution
    N_largest_freqs = 10
    fourier_transform = np.abs(np.real(np.fft.rfft2(arr[0,0])))
    fourier_transform[0,0] = 0  # Remove the measurement of the average (freq 0 component)

    # Find largest frequency components
    l_freq_idx = np.argsort(fourier_transform, axis=None)
    x_freqs = [i % fourier_transform.shape[0] for i in l_freq_idx]
    y_freqs = [i // fourier_transform.shape[1] for i in l_freq_idx]

    # Only keep ones that describe half wavelengths of 
    largest_x_freqs = [f for f in x_freqs if f < fourier_transform.shape[0] // 2][-N_largest_freqs:]
    largest_y_freqs = [f for f in y_freqs if f < fourier_transform.shape[1] // 2][-N_largest_freqs:]
    largest_frequencies = fourier_transform[largest_x_freqs, largest_y_freqs]
    print(largest_x_freqs, largest_y_freqs)

    plt.title("Fourier Transform (real)")
    plt.scatter(largest_x_freqs, largest_y_freqs, c=largest_frequencies, cmap=plt.cm.winter)
    plt.colorbar()
    plt.show()
