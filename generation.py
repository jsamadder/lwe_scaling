import numpy as np
import math
from abc import ABC, abstractmethod
from scipy.stats import dlaplace

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

    @abstractmethod
    def get_mean(self):
        pass

    def get_variance(self):
        pass

    def get_skew(self):
        pass

    def get_kurtosis(self):
        pass

class DiscreteGaussian(Distribution):
    """
    Discrete Guassian distribution

    q - modulus
    mean - mean
    std - standard deviation
    lim - bound for samples
          values outside [-lim*sigma, lim*sigma] have probability 0
    """

    def __init__(self, q: int, mean: int=0, std: float=1.0, lim: float=6.0):
        self.q = q
        self.mean = mean
        self.std = std
        self.bound = math.floor(lim * std)
        self.fmax = self.pdf(0) # 0 has highest probability
        self.ptable = [self.pdf(x) for x in range(-self.bound, self.bound+1)]

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.std ** 2

    def get_skew(self):
        return 0

    def get_kurtosis(self):
        return 3

    def sample(self, m):
        samples = []
        for _ in range(m):
            samples.append(self.rejection_sample())
        return np.array(samples, dtype=np.int64)

    def pdf(self, x):
        """
        pdf for Gaussian dist
        """

        coeff = 1/(np.sqrt(2 * np.pi * (self.std ** 2)))
        num = (x - self.mean) ** 2
        denom = 2 * (self.std ** 2)
        exponent = np.exp(-num/denom)
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

class DiscreteLaplacian(Distribution):
    """
    Discrete Laplacian distribution

    q: modulus
    a: scale parameter
    loc: location parameter
    """

    def __init__(self, q: int, a: float=1.0, loc: int=0):
        self.q = q
        self.a = a
        self.loc = loc
        self.mean, self.variance, self.skew, self.kurtosis = dlaplace.stats(a, loc=loc, moments='mvsk')
    

    def sample(self, m):
        samples = []
        for _ in range(m):
            samples.append(dlaplace.rvs(self.a, loc=0) % self.q)
        return np.array(samples, dtype=np.int64)

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.variance

    def get_skew(self):
        return self.skew

    def get_kurtosis(self):
        return self.kurtosis

class SparseErrors(Distribution):
    """
    Create a sparse error vector

    q: modulus
    k: number of non-zero entries of error vector
    """

    def __init__(self, q: int, k: int):
        self.q = q
        self.k = k

    def sample(self, m):
        error = np.zeros((m,1), dtype=np.int64)
        sparse_indices = np.random.choice(m, size=self.k, replace=False)
        error[sparse_indices] = np.random.choice(self.q, size=(self.k,1))

        return error