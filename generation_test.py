import generation
import time

n = 10
q = 37

def time_test(num_samples, non_zeros):
    """
    Test how long it takes to generate LWE samples

    - num_samples: how many LWE samples to create for each error distribution
    - non_zeros: how many non_zeros to have for sparse errors distribution
    """

    start_time = time.perf_counter()
    dists = [
        generation.DiscreteGaussian(q=q),
        generation.DiscreteLaplacian(q=q),
        generation.SparseErrors(q=q, k=sparse_non_zeros),
        generation.CenteredBinomial(q=q, eta=10),
    ]

    generator = generation.LWEGenerator(n, q, dists)
    generator.generate_samples(samples)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Generating {samples} samples took {total_time:.3f} seconds")


samples = 1_000_000
sparse_non_zeros = 100
time_test(samples, sparse_non_zeros)

