import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Belousov():
    def __init__(self, alpha,beta,gamma,gridsize,t):
        self.nx = gridsize
        self.ny = gridsize
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.t = t
        self.A = []
        self.B = []
        self.C = []
        self.initial()
        self.run()

    def initial(self):
        # Initialize the array with random amounts of A, B, and C.
        self.grid = np.random.random(size=(2, 3, self.ny, self.nx))
        self.grid[:, 1] = 0.5
        

    def update(self, p, arr):
        """Update arr[p] to arr[q] by evolving in time."""

        q_index = (p + 1) % 2
        new_arr = np.zeros((3, self.ny, self.nx))
        kernel = np.ones((3, 3)) / 9

        for component in range(3):
            for y in range(self.ny):
                for x in range(self.nx):
                    total = 0
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            total += arr[p, component, (y + i) % self.ny, (x + j) % self.nx]
                    new_arr[component, y, x] = total / 9

        arr[q_index, 0] = new_arr[0] + new_arr[0] * (self.alpha * new_arr[1] - self.gamma * new_arr[2])
        arr[q_index, 1] = new_arr[1] + new_arr[1] * (self.beta * new_arr[2] - self.alpha * new_arr[0])
        arr[q_index, 2] = new_arr[2] + new_arr[2] * (self.gamma * new_arr[0] - self.beta * new_arr[1])
        np.clip(arr[q_index], 0, 1, arr[q_index])
        return arr

    def run(self):
        for i in range(self.t):
            self.grid = self.update(i % 2, self.grid)
            self.A.append(np.sum(self.grid[0,0]))
            self.B.append(np.sum(self.grid[0,1]))
            self.C.append(np.sum(self.grid[0,2]))

        return self.grid
        
    def plot(self):
        fig, ax = plt.subplots()
        im = ax.imshow(self.grid[0, 0], cmap=plt.cm.winter)
        im.set_array(self.grid[self.t % 2, 0])
        ax.axis('off')
        plt.show()

    def animate(self):
        fig, ax = plt.subplots()
        im = ax.imshow(self.grid[0, 0], cmap=plt.cm.winter)
        ax.axis('off')
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel('Concentration A')

        def update_frame(frame):
            self.grid = self.update(frame % 2, self.grid)
            im.set_array(self.grid[frame % 2, 0])
            return [im]

        ani = animation.FuncAnimation(fig, update_frame, frames=self.t, interval=50, blit=True)
        plt.show()

    def concentration(self):
        plt.plot(np.linspace(0,self.t,self.t),self.A)
        plt.show()


bel2 = Belousov(1,1,1,250,100)
bel2.animate()