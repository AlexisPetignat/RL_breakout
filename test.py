import matplotlib.pyplot as plt
import numpy as np

N = 7
E = 2000
l = np.array([i * (N - e) / N for e in range(N) for i in range(E, 500, -1)]) / E

plt.plot(l)

plt.title("Evolution of Epsilon with epochs")
plt.savefig("test.png")
