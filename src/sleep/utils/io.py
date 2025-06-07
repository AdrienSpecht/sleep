from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


def normalise_inputs(
    key: str | int | Sequence | np.ndarray | pd.Series | torch.Tensor,
    time: float | Sequence | np.ndarray | pd.Series | torch.Tensor,
    params: dict[str, Any],
    key_id: dict[str | int | Sequence | np.ndarray | pd.Series | torch.Tensor, int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Normalise and broadcast arguments."""
    # ── 1 · keys ───────────────────────────────────────────
    key_arr = to_flat_cpu_array(key)
    unknown = np.setdiff1d(key_arr, list(key_id.keys()))
    if unknown.size:
        raise ValueError(f"Unknown key(s): {unknown}")

    id_vector = np.vectorize(key_id.__getitem__)(key_arr)
    id_tensor = to_tensor(id_vector, device, dtype=torch.long)
    n = id_tensor.numel()

    # ── 2 · time ───────────────────────────────
    time_tensor = to_tensor(time, device, dtype)

    if time_tensor.ndim == 0:  # scalar → (n,1)
        time_tensor = time_tensor.expand(n, 1)
    elif time_tensor.ndim == 1:  # (m,) vector
        m = time_tensor.numel()
        time_tensor = time_tensor.expand(n, m)  # (n,m)
    elif time_tensor.ndim == 2:  # (nt,m) matrix
        nt, m = time_tensor.shape
        if nt not in (1, n):
            raise ValueError(f"{nt} rows in `time` but {n} keys supplied.")
        if nt == 1 and n > 1:  # broadcast rows
            time_tensor = time_tensor.expand(n, m)
    else:
        raise ValueError("`time` must be 0, 1 or 2-D.")

    # ── 3 · params ─────────────────────
    # Convert and find the number of configurations to compute p
    p = 1
    param_tensors = {}
    for name, value in params.items():
        tensor = to_tensor(value, device, dtype)
        if tensor.ndim == 2 and tensor.shape[1] != 1:
            p = max(p, tensor.shape[1])
        elif tensor.ndim == 1 and tensor.numel() not in (1, n):
            p = max(p, tensor.numel())
        param_tensors[name] = tensor

    # Broadcast each parameter
    for name, tensor in param_tensors.items():
        if tensor.ndim == 0:  # scalar
            param_tensors[name] = tensor.expand(n, p)
        elif tensor.ndim == 1:
            nelem = tensor.numel()
            if nelem == 1:  # (1,)
                param_tensors[name] = tensor.expand(n, p)
            elif nelem == n:  # (n,)
                param_tensors[name] = tensor.unsqueeze(1).expand(n, p)
            elif nelem == p:  # (p,)
                param_tensors[name] = tensor.expand(n, p)
            else:
                raise ValueError(f"Cannot broadcast {name} of shape {(nelem,)} to ({n},{p})")
        elif tensor.ndim == 2:
            r, c = tensor.shape
            if (r, c) == (1, p) or (r, c) == (n, 1):
                param_tensors[name] = tensor.expand(n, p)
            elif (r, c) == (n, p):
                param_tensors[name] = tensor
            else:
                raise ValueError(f"Cannot broadcast {name} of shape ({r},{c}) to ({n},{p})")

    return id_tensor, time_tensor, param_tensors


def to_flat_cpu_array(obj: Any) -> np.ndarray:
    """
    Turn *obj* into a 1-D Numpy array **on CPU**.
    Used only for *keys*  → always small, copy cost negligible.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().ravel()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.to_numpy().ravel()
    return np.asarray(obj).ravel()  # list / tuple / numpy / scalar


def to_tensor(obj: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert *obj* to a torch.Tensor on *device* with *dtype*."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, dtype=dtype)
    if isinstance(obj, (pd.Series, pd.Index)):
        obj = obj.to_numpy()
    return torch.as_tensor(obj, dtype=dtype, device=device)
