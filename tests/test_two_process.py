import numpy as np
import pandas as pd
import pytest

from sleep import Model


@pytest.fixture(name="diary")
def fixture_diary():
    return pd.read_csv("data/diary.csv")


def test_two_process(diary):
    """
    Test the Model.compute of the Two-Process model by verifying that the numerical solution matches the expected
    differential equations for homeostasis:

    Differential equations:
        During wake:
            ds/dt = (1 - s) / t_w
        During sleep:
            ds/dt = -s / t_s
        where:
            - s: homeostasis
            - t_w: time constant during wake
            - t_s: time constant during sleep

            Supplementary parameters for the periodic solution:
            - T: period
            - need: sleep need
    """
    model = Model(name="two_process", diary=diary)
    key = "Chronic Sleep Restriction: 21h / 3h → Recovery: 16h / 8h"
    res = 10  # min
    rtol = 0.1
    time = np.arange(0, 10 * 24 * 60, res)  # every 10 min
    time = time / 60  # min -> h
    out = model.compute(key, time)  # (1000, )
    s = out["homeostasis"]
    t_s = model.defaults["t_s"]
    t_w = model.defaults["t_w"]
    T = model.defaults["T"]
    need = model.defaults["need"]

    # Identify wake/sleep segments
    sleep_mask = np.zeros_like(time, dtype=bool)
    cd_key = diary["key"] == key
    for awake, asleep in zip(diary[cd_key]["awake"], diary[cd_key]["asleep"]):
        sleep_mask[(time >= asleep) & (time < awake)] = True
    wake_mask = ~sleep_mask
    print("sleep_mask:", sleep_mask)

    # Compute ds/dt and d(d)/dt
    ds_dt = np.diff(s) / np.diff(time)
    s_mid = (s[1:] + s[:-1]) / 2

    # During wake: ds/dt = (1-s)/t_w
    expected_ds_dt = (1 - s_mid[wake_mask[:-1]]) / t_w
    np.testing.assert_allclose(ds_dt[wake_mask[:-1]], expected_ds_dt, rtol=rtol)

    # During sleep: ds/dt = -(s)/t_s
    expected_ds_dt = -(s_mid[sleep_mask[:-1]]) / t_s
    np.testing.assert_allclose(ds_dt[sleep_mask[:-1]], expected_ds_dt, rtol=rtol)
