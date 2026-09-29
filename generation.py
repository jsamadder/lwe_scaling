import numpy as np
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

    def genereate_samples(self, m: int, pos: bool=True):
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

    @abstractmethod
    def variance(self):
        pass

class DiscreteGaussian(Distribution):
    """
    Discrete Guassian distribution
    """
    def __init__(self, sigma: float, q: int):
        self.sigma = sigma
        self.q = q

    def sample(self, m):
        

