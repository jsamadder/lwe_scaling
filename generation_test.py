import generation
import time


def time_test(num_samples, m, non_zeros):
    """
    Test how long it takes to generate LWE samples

    - num_samples: how many LWE samples to create for each error distribution (randomized A,s
                   for each sample)
    - m: number of equations per sample
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
    gen_samples = generator.generate_samples(num_samples=samples, m=m)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Generating {samples} LWE samples with {m} equations and {len(dists)} distributions took {total_time:.3f} seconds")

    print(gen_samples["DiscreteGaussian(q=37, mean=0, std=1.0, lim=6.0)"])

n = 10
q = 37
m = n
samples = 1000
sparse_non_zeros = 3
time_test(samples, m, sparse_non_zeros)

