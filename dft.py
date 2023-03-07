import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-poster')

sampling_rate = 100.0
sampling_interval = 1.0 / sampling_rate
t = np.arange(0, 1, sampling_interval)

frequency = 1 # Hz
x = 3 * np.sin(2 *  np.pi * frequency * t)

frequency = 2 # Hz
x += 3 * np.sin(2 *  np.pi * frequency * t)

frequency = 4 # Hz
x += 3 * np.sin(2 *  np.pi * frequency * t)

frequency = 8 # Hz
x += 3 * np.sin(2 *  np.pi * frequency * t)

plt.plot(t, x)
plt.xlabel("Time")
plt.ylabel("Signal")
plt.show()

def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)

    return X

X = DFT(x)
N = len(x)
n = np.arange(N)
T = N / sampling_rate
frequency = n / T
plt.stem(frequency, abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel("Frequency (Hz)")
plt.ylabel("DFT Amplitude")
plt.show()
