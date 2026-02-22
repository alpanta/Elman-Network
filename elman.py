import numpy as np
import matplotlib.pyplot as plt


def acti_func(value):
    """Activation function (tanh) and its derivative."""
    a = np.tanh(value)
    der = 1 - a ** 2
    return a, der


# Initial conditions
y = {0: 0.1, 1: 0.1}

index = 0  # Variable for plotting test data
sw = 2  # 1 = random weights, 2 = nearly best (recorded) weights

nnu = 1  # Input layer (y(k-1))
nnx = 6  # Hidden layer
nny = 1  # Output layer

training_set_size = 900  # Training data (first 900 elements of Billings System)
test_set_size = training_set_size + 40  # Test data

iteration = 15
momentum = 0.65

if sw == 1:
    n = 0.029  # Learning rate
    weights_u = np.random.randn(nnx, nnu)
    weights_x = np.random.randn(nnx, nnx)
    weights_y = np.random.randn(nny, nnx)

if sw == 2:
    n = 0.027  # Learning rate

    # Nearly best initial conditions for weights (Recorded weights)
    weights_x = np.array([
        [ 0.8908,  1.8106,  1.6085, -1.2210, -1.0242,  0.5656],
        [-1.8913,  0.5387, -0.5894, -0.0360,  0.4159, -1.8232],
        [ 1.1209,  1.0958, -1.2613,  0.5687, -2.3102, -3.7303],
        [-0.3719, -0.8632, -0.4843, -0.6312, -0.2901,  1.6408],
        [-1.5000, -1.7704, -0.1812,  0.9253, -0.6058,  1.5762],
        [ 0.8455,  1.8453,  0.9314,  0.9002, -1.9390,  0.2969],
    ])

    weights_u = np.array([
        [-0.8333],
        [ 0.2984],
        [-0.4605],
        [ 0.4958],
        [-1.3770],
        [ 0.0835],
    ])

    weights_y = np.array([
        [0.1797, 0.0751, 0.1256, 0.0727, -0.3476, 0.3671],
    ])

# Weights to hold previous values for momentum term
wxold = np.zeros((nnx, nnx))
wuold = np.zeros((nnx, nnu))
wyold = np.zeros((nny, nnx))

xold = np.zeros((nnx, 1))  # Previous hidden layer output

y_network = {}

decay_rate = 0.01  # Learning rate decay coefficient

# Training
for i in range(iteration):
    lr = n / (1 + decay_rate * i)  # Decaying learning rate

    # Reset state at the start of each epoch
    y = {0: 0.1, 1: 0.1}
    y_network = {}
    xold = np.zeros((nnx, 1))
    epoch_error = 0.0

    for k in range(2, training_set_size):  # MATLAB 3:900 → Python 2:899
        inp = np.random.normal(0, 0.01)  # Noise implemented for input

        # Billings System
        y[k] = ((0.8 - 0.5 * np.exp(-y[k - 1] ** 2)) * y[k - 1]
                - (0.3 + 0.9 * np.exp(-y[k - 1] ** 2)) * y[k - 2]
                + 0.1 * np.sin(np.pi * y[k - 1]) + inp)

        # Feedforward — use y(k-1) as input instead of noise to eliminate phase lag
        u = np.array([[y[k - 1]]])
        v = weights_u @ u + weights_x @ xold

        x, f_der = acti_func(v)
        xold = x

        y_network[k] = (weights_y @ x)[0, 0]

        e = y[k] - y_network[k]  # Error
        epoch_error += e ** 2

        # Store previous weights for momentum
        tempx = weights_x.copy()
        tempu = weights_u.copy()
        tempy = weights_y.copy()

        # Weight updates
        weights_x = weights_x + lr * (weights_y.T * e) * f_der @ x.T + momentum * (weights_x - wxold)
        weights_u = weights_u + lr * (weights_y.T * e) * f_der @ u.T + momentum * (weights_u - wuold)
        weights_y = weights_y + lr * e * x.T + momentum * (weights_y - wyold)

        wxold = tempx
        wuold = tempu
        wyold = tempy

    mse = epoch_error / (training_set_size - 2)
    print(f"Epoch {i + 1:2d}/{iteration} | LR: {lr:.5f} | MSE: {mse:.6f}")

# Test
y1 = []
y2 = []

for k in range(training_set_size - 1, test_set_size):  # MATLAB training_set_size:test_set_size
    inp = np.random.normal(0, 0.01)

    y[k] = ((0.8 - 0.5 * np.exp(-y[k - 1] ** 2)) * y[k - 1]
            - (0.3 + 0.9 * np.exp(-y[k - 1] ** 2)) * y[k - 2]
            + 0.1 * np.sin(np.pi * y[k - 1]) + np.random.normal(0, 0.01))

    y1.append(y[k])

    u = np.array([[y[k - 1]]])
    v = weights_u @ u + weights_x @ xold

    x, f_der = acti_func(v)
    xold = x

    y_network[k] = (weights_y @ x)[0, 0]
    y2.append(y_network[k])

# Plot 1: First 150 elements of Billings System
plt.figure(1)
y_vals = [y[i] for i in range(min(len(y), 150))]
plt.plot(y_vals)
plt.title("Billings System")
plt.axis([0, 150, -1.5, 1.5])

# Plot 2: System Outputs vs Network Outputs
plt.figure(2)
plt.plot(y1, "-*r", label="System Outputs")
plt.plot(y2, "-ob", label="Network Outputs")
plt.title("System Outputs vs. Network Outputs")
plt.legend()

plt.show()
