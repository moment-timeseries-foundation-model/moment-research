import numpy as np

from .base import FPTDataset


class BitMemoryDataset(FPTDataset):
    def __init__(self, n=1000, num_patterns=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.num_patterns = num_patterns

    def get_batch_np(self, batch_size, **kwargs):
        bits = np.random.randint(
            low=0, high=2, size=(batch_size, self.num_patterns, self.n)
        )
        bits = 2 * bits - 1
        query_inds = np.random.randint(low=0, high=self.num_patterns, size=batch_size)
        query_bits = bits[range(batch_size), query_inds]
        mask = np.random.randint(low=0, high=2, size=query_bits.shape)
        masked_query_bits = mask * query_bits
        masked_query_bits = masked_query_bits.reshape(batch_size, 1, self.n)
        x = np.concatenate([bits, masked_query_bits], axis=1)
        y = query_bits
        return x, y


class BitXORDataset(FPTDataset):
    def __init__(self, n=5, num_patterns=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.num_patterns = num_patterns

    def get_batch_np(self, batch_size, **kwargs):
        bits = np.random.randint(
            low=0, high=2, size=(batch_size, self.num_patterns, self.n)
        )
        xored_bits = bits[:, 0]
        for i in range(1, self.num_patterns):
            xored_bits = np.logical_xor(xored_bits, bits[:, i])
        return bits, xored_bits
