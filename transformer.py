import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from generation import LWEGenerator, Distribution, DiscreteGaussian, CenteredBinomial, SparseErrors

class LWEDataset(Dataset):
    """
    Takes tuples returned by LWEGenerator.generate_samples and formats it to the torch Dataset object
    """

    def __init__(self, samples, distribution: Distribution):
        self.items = []
        for sample in samples:
            