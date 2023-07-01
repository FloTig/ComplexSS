import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import convolve2d
from multiscale_complexity import calc_complexity
import fourier_analysis as fa

from os import mkdir, path

class SimpleBZ:
    def __init__(self, nx, ny, alpha, gamma, beta, interval, init_arr=None):
        if init_arr is not None:
            assert ny == init_arr.shape[0]
            assert nx == init_arr.shape[1]
            self.arr = init_arr
        else:
            self.arr = np.random.random(size=(2, 3, ny, nx))
        self.nx = nx
        self.ny = ny

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.i = 0
        self.measure_interval = interval
        self.c_lists = [[], [], []]
        self.cmplx_list = []


    def update(self, i):
        """Updates the state of the BZ reaction after one timestep"""
        self.i += 1

        # Count the average amount of each species in the 9 cells around each cell
        # by convolution with the 3x3 array m.
        p = self.i % 2
        q = (p+1) % 2
        s = np.zeros((3, self.ny, self.nx))
        neighbor_kernel = np.ones((3,3)) / 9
        for k in range(3):
            s[k] = convolve2d(self.arr[p,k], neighbor_kernel, mode='same', boundary='wrap')

        # Append concentrations
        for k, c_list in enumerate(self.c_lists):
            c_list.append(self.arr[p,k].sum() / (nx * ny))

        # Calculate expensive metrics
        if (self.i-1) % self.measure_interval == 0:
            self.cmplx_list.append(calc_complexity(self.arr[p,0]))

        # Apply the reaction equations
        self.arr[q,0] = s[0] + s[0]*(self.alpha * s[1] - self.gamma * s[2])
        self.arr[q,1] = s[1] + s[1]*(self.beta  * s[2] - self.alpha * s[0])
        self.arr[q,2] = s[2] + s[2]*(self.gamma * s[0] - self.beta  * s[1])

        # Ensure the species concentrations are kept within [0,1].
        np.clip(self.arr[q], 0, 1, self.arr[q])


    def evolve(self, Nsteps):
        for i in range(Nsteps):
            self.update(i)


    def animate(self, Nsteps):
        # Set up the image
        fig, (ax1, ax2) = plt.subplots(2,1)
        im = ax1.imshow(self.arr[0,0], cmap=plt.cm.winter)
        ax1.axis('off')

        line1, = ax2.plot([0], [0])
        # line2, = ax2.plot([0], [0])
        # line3, = ax2.plot([0], [0])

        def next_frame(i):
            """Update the image for iteration i of the Matplotlib animation."""

            self.update(i)

            # Plot the concentration distribution
            im.set_array(self.arr[i % 2, 0])

            # Plot all three average concentrations against time
            # ax2.set_xlim(0, len(self.c_lists[0]))
            # ax2.set_ylim(0.9 * min(min(c_list) for c_list in self.c_lists),
            #              1.1 * max(max(c_list) for c_list in self.c_lists))
            # ax2.figure.canvas.draw()
            #
            # x = list(range(len(self.c_lists[0])))
            # line1.set_data(x, self.c_lists[0])
            # line2.set_data(x, self.c_lists[1])
            # line3.set_data(x, self.c_lists[2])

            # Plot Complexity
            ax2.set_xlim(0, Nsteps//self.measure_interval)
            ax2.set_ylim(0, 20)
            ax2.figure.canvas.draw()

            x = list(range(len(self.cmplx_list)))
            line1.set_data(x, self.cmplx_list)

            return [im, line1]  # , line2, line3]

        animation.FuncAnimation(fig, next_frame, frames=Nsteps, interval=1, blit=True)

        # To view the animation, uncomment this line
        plt.show()


    def plot_frame(self, total_step, frame_num, folder_name):
        fig, (ax2, ax1) = plt.subplots(1,2, figsize=(12, 6))
        fig.suptitle(f"Rates {alpha:.4f} {beta:.4f} {gamma:.4f}")

        ax1.imshow(self.arr[0,0], cmap=plt.cm.winter)

        x = list(range(len(self.cmplx_list)))
        ax2.plot(x, self.cmplx_list)
        ax2.set_xlabel(f"Timestep / {self.measure_interval}")
        ax2.set_ylabel("Complexity")
        ax2.set_xlim(0, Nsteps//self.measure_interval)

        plt.savefig(folder_name + path.sep + str(frame_num) + ".png")
        plt.close(fig)


    def plot_concentrations(self, ax):
        # Show plot of all concentrations over time
        ax.set_title("Total Concentrations")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Concentration")
        for i, c_list in enumerate(self.c_lists):
            ax.plot(c_list, label=str(i))
        ax.legend()

    def plot_2Dconcentration(self, ax, i_concentration=0):
        # Plot the final concentration distribution
        ax.set_title("Concentration of substance 1")
        ax.imshow(self.arr[0, i_concentration], cmap=plt.cm.winter)

    def plot_3D_attractor(self):
        # 3D attractor plot
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(*self.c_lists)
        plt.show()


if __name__ == '__main__':
    # Width, height of the image.
    nx, ny = 256, 256
    basis_rate = 0.5

    EXPLORING = False
    LIVE = True
    SAVE_FRAMES = True
    Nsteps = 2000
    factors = [0.5]  # , 0.75, 1, 1.25, 1.5]

    # Loop over different combinations of reaction rates.
    for factor in factors:
        print(factor)

        interval = 20
        alpha, beta, gamma = basis_rate, basis_rate, basis_rate * factor
        sim = SimpleBZ(nx, ny, alpha, beta, gamma, interval)

        if EXPLORING:
            if LIVE:
                sim.animate(Nsteps)
            else:
                sim.evolve(Nsteps)

                # 2D fourier stuff
                # Plot frequency information next to the final image
                fig, (imax, ax1, ax2) = plt.subplots(1,3, figsize=(18, 6))
                sim.plot_2Dconcentration(imax)

                freq_points = fa.plot_freqs(sim.arr[0, 0], ax1)
                fig.colorbar(freq_points, cmap=plt.cm.winter)
                fa.plot_lengths(sim.arr[0, 0], ax2)
                plt.show()

                plt.title("Multiscale Complexity")
                plt.plot(sim.cmplx_list)
                plt.show()

        else:
            folder_name = f"./x{nx}-y{ny}_a{alpha:.5f}-b{beta:.5f}-g{gamma:.5f}_N{Nsteps}_int{interval}"

            if SAVE_FRAMES:
                mkdir(folder_name)
                for frame in range(Nsteps//interval):
                    sim.evolve(interval)
                    sim.plot_frame(Nsteps, frame, folder_name)

            else:
                sim.evolve(Nsteps)

                # Plot final state of the simulation
                fig, [ax1, ax2] = plt.subplots(1,2, figsize=(12,6))
                fig.suptitle(f"Rates {alpha:.4f} {beta:.4f} {gamma:.4f}")
                sim.plot_concentrations(ax1)
                sim.plot_2Dconcentration(ax2)
                plt.savefig(folder_name + ".png")

                plt.show()


        # Print (relative) reaction rates and final concentrations
        print(alpha, beta, gamma)
        print([f'{c / max(alpha, beta, gamma):.2f}' for c in [alpha, beta, gamma]])
        final_cs = [c[-1] for c in sim.c_lists]
        print(final_cs)
        print([f'{c / max(final_cs):.2f}' for c in final_cs])
