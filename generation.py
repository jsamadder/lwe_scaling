import numpy as np
import math
from abc import ABC, abstractmethod

class LWEGenerator():
    """
    Generate LWE samples, i.e create mxn matrix A and b = A*s + e (mod q)
     - s is the secret
     - e is the error
     - q is the prime modulus
    """
    
    def __init__(self, n: int, q: int, dist: Distribution):
        """
        n: dimension
        q: prime modulus
        e: error distribution
        """
        self.n = n
        self.q = q
        self.dist = dist

    def generate_samples(self, m: int, pos: bool=True):
        """
        pos: b = (A @ s + e) % q
        otherwise b is sampled uniformly, no LWE structure
        """
        A = np.random.randint(0, self.q, size=(m, self.n))
        s = np.random.randint(0, self.q, size=(self.n,1))
        e = self.dist.sample(m)

        if pos:
            b = (A @ s + e) % self.q
            label = 1
        else:
            b = np.random.randint(0, self.q, size=(m,1))
            label = 0
        return A, b, label

class Distribution(ABC):
    """
    Abstract class for child distribution classes (discrete Gaussian, rounded Gaussian, etc.) to
    implement
    """
    @abstractmethod
    def sample(self, m: int) -> np.ndarray:
        pass

class DiscreteGaussian(Distribution):
    """
    Discrete Guassian distribution

    q - modulus
    sigma - standard deviation
    lim - bound for samples
          values outside [-lim*sigma, lim*sigma] have probability 0
    """

    def __init__(self, q: int, sigma: float = 1.0, lim: float=6.0):
        self.sigma = sigma
        self.q = q
        self.bound = math.floor(lim * sigma)
        self.fmax = self.pdf(0) # 0 has highest probability
        self.ptable = [self.pdf(x) for x in range(-self.bound, self.bound+1)]

    def sample(self, m):
        samples = []
        for _ in range(m):
            samples.append(self.rejection_sample())
        return np.array(samples, dtype=np.int64)

    def pdf(self, x):
        """
        pdf for Gaussian dist
        """

        coeff = 1/(self.sigma * np.sqrt(2 * np.pi))
        exponent = np.exp(-0.5 * (x / self.sigma) ** 2)
        return coeff * exponent

    def rejection_sample(self):
        """
        Return single sample in range [0, q-1]
        """
        while True:
            x = np.random.randint(-self.bound, self.bound+1)
            y = np.random.random() * self.fmax
            if y > self.ptable[x + self.bound]:
                continue
            else:
                break
        return x % self.q

    