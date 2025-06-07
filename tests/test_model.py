import numpy as np
import pandas as pd
import pytest
import torch

from sleep import Model


@pytest.fixture(name="diary")
def fixture_diary():
    return pd.read_csv("tutorial/diary.csv")


def test_model_init(diary):
    with pytest.raises(ValueError):
        Model(name="not_supported_model", diary=diary)
    model = Model(name="unified", diary=diary)
    assert model.name == "unified"


def test_model_io(diary):
    model = Model(name="unified", diary=diary)
    wrong_key = 1
    assert wrong_key not in model.key_id
    with pytest.raises(ValueError):
        model.compute(key=wrong_key, time=1)

    # 0-D
    key = list(model.key_id.keys())[0]
    time = 1.0
    out = model.compute(key=key, time=time)
    assert list(out.keys()) == ["homeostasis", "debt"]
    assert out["homeostasis"].shape == ()
    # Compatible with Sequence, array, series, tensor
    model.compute([key], time)
    model.compute(np.asarray([key]), time)
    model.compute([key], np.asarray([time]))
    model.compute(key, pd.Series([time]))
    time_tensor = torch.as_tensor([time])
    time_tensor.requires_grad = True
    out = model.compute(key, time_tensor)
    assert out["homeostasis"].shape == ()
    out["homeostasis"].backward()
    assert time_tensor.grad is not None

    # Parameters
    uncomplete_params = {"T": 24.0, "need": 8.0, "t_w": 40.0, "t_s": 1.0}
    model.compute(key, time, **uncomplete_params)
    wrong_params = {"T": 0.0, "need": 8.0, "t_w": 40.0, "t_s": 1.0, "t_la": 97.4}  # T = 0
    unrecognized_params = {"T": 24.0, "need": 8.0, "t_w": 40.0, "t_s": 1.0, "unrecognized": 1.0}
    with pytest.raises(ValueError):
        model.compute(key, time, **wrong_params)
        model.compute(key, time, **unrecognized_params)

    # keys = n, params = p
    # if param.squeeze() is 1-D, it should be of dimension (n, ), or (p, )
    # if param.squeeze() is 2-D, it should be of dimension (n, p).
    params = {"T": 2 * [24.0], "need": 3 * [8.0]}
    out = model.compute(2 * [key], time, **params)
    assert out["homeostasis"].shape == (2, 3)
    out = model.compute(3 * [key], time, **params)
    assert out["homeostasis"].shape == (3, 2)
    with pytest.raises(ValueError):  # p1 != p2
        model.compute(key, time, **params)
        model.compute(4 * [key], time, **params)
        params["t_w"] = 4 * [40.0]
        model.compute(2 * [key], time, **params)

    # 1-D
    out = model.compute(key, time, T=[23.0, 24.0])  # T=(2,)
    assert out["homeostasis"].shape == (2,)
    out = model.compute(key, 2 * [time])  # time=(2,)
    assert out["homeostasis"].shape == (2,)
    out = model.compute(key, [2 * [time]])  # time=(1,2)
    assert out["homeostasis"].shape == (2,)

    # 2-D
    out = model.compute(3 * [key], time, T=[23.0, 24.0])  # key=(3,), params=(2,)
    assert out["homeostasis"].shape == (3, 2)
    out = model.compute(3 * [key], 2 * [time])  # key=(3,), time=(2,)
    assert out["homeostasis"].shape == (3, 2)
    out = model.compute(3 * [key], [2 * [time]])  # key=(3,), time=(1,2)
    assert out["homeostasis"].shape == (3, 2)
    out = model.compute(3 * [[key]], 2 * [time])  # key=(3,1), time=(2,)
    assert out["homeostasis"].shape == (3, 2)
    out = model.compute(3 * [[key]], [2 * [time]])  # key=(3,1), time=(1,2)
    assert out["homeostasis"].shape == (3, 2)

    # 3-D
    out = model.compute(3 * [key], 4 * [time], T=[23.0, 24.0])  # key=(3,), time=(4,), params=(2,)
    assert out["homeostasis"].shape == (3, 2, 4)
