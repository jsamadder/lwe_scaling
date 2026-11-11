import generation
import time

n = 10
q = 37
samples = 1_000_000_000
sparse_non_zeros = 100

start_time = time.perf_counter()
dists = [
    generation.DiscreteGaussian(q=q),
    generation.DiscreteLaplacian(q=q),
    generation.SparseErrors(q=q, k=sparse_non_zeros),
    generation.CenteredBinomial(q=q, eta=10),
]

generator = generation.LWEGenerator(n, q, dists)

end_time = time.perf_counter()
total_time = end_time - start_time
print(f"Generating {samples} samples took {total_time:.3f} seconds")

A, s, errors, bs, labels = generator.generate_samples(1000)

"""
things = errors["SparseErrors"]
non_zeros = 0
for thing in things:
    if thing.item() != 0:
        non_zeros += 1
print(f"Desired non-zeros: {sparse_non_zeros}")
print(f"Actual non-zeros: {non_zeros}")
"""