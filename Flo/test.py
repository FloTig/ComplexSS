import numpy as np
import matplotlib.pyplot as plt

# Width, height of the image.
nx, ny = 250, 250
# Reaction parameters.
alpha, beta, gamma = 1.2, 1, 1

def update(p, arr):
    """Update arr[p] to arr[q] by evolving in time."""

    q = (p+1) % 2
    s = np.zeros((3, ny, nx))
    m = np.ones((3, 3)) / 9

    for k in range(3):
        for y in range(ny):
            for x in range(nx):
                total = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        total += arr[p, k, (y + i) % ny, (x + j) % nx]
                s[k, y, x] = total / 9

    arr[q, 0] = s[0] + s[0] * (alpha * s[1] - gamma * s[2])
    arr[q, 1] = s[1] + s[1] * (beta * s[2] - alpha * s[0])
    arr[q, 2] = s[2] + s[2] * (gamma * s[0] - beta * s[1])
    np.clip(arr[q], 0, 1, arr[q])
    return arr

# Initialize the array with random amounts of A, B, and C.
arr = np.random.random(size=(2, 3, ny, nx))


num_frames = 200
for i in range(num_frames):
    print(np.shape(arr))
    arr = update(i % 2, arr)
    

# Set up the image
fig, ax = plt.subplots()
im = ax.imshow(arr[0, 0], cmap=plt.cm.winter)
im.set_array(arr[i % 2, 0])
ax.axis('off')

plt.show()
