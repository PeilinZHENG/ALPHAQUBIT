import os
import tempfile
import numpy as np
import torch
import pytest

from ai_models.model import PauliPlusDataset


def test_pauli_plus_dataset_autopad_and_mask():
    N, R, S = 5, 2, 5  # S+1=6 -> not perfect square
    with tempfile.TemporaryDirectory() as tmpdir:
        synd_path = os.path.join(tmpdir, "synd.npy")
        log_path = os.path.join(tmpdir, "log.npy")
        np.save(synd_path, np.random.randint(0, 2, size=(N, R, S)))
        np.save(log_path, np.random.randint(0, 2, size=(N,)))

        ds = PauliPlusDataset(synd_path, log_path, basis_id=0)
        (x, basis, mask), y = ds[0]

        d = int(np.ceil(np.sqrt(S + 1)))  # expected grid size
        expected_S = d * d - 1
        assert x.shape[1] == expected_S
        assert mask.shape[0] == expected_S

        # compute expected mask pattern
        expected_mask = []
        for i in range(1, expected_S + 1):
            r, c = divmod(i, d)
            expected_mask.append(1 if (r + c) % 2 == 0 else 2)
        expected_mask = torch.tensor(expected_mask, dtype=torch.long)
        assert torch.equal(mask, expected_mask)
        assert basis.item() == 0
        assert isinstance(y.item(), float)
