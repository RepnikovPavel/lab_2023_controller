import numpy as np
from typing import Tuple

def make_batches(vector_of_numbers: np.array, num_of_batches: int) -> Tuple[np.array, np.array]:
    N = len(vector_of_numbers)
    batch_size = N // num_of_batches
    batches = np.zeros(shape=(num_of_batches, batch_size), dtype=np.intc)
    for i in range(num_of_batches):
        batches[i] = vector_of_numbers[i * batch_size:(i + 1) * batch_size]
    rest = vector_of_numbers[num_of_batches * batch_size:]
    return batches, rest