import numpy as np
import pandas as pd
import pytest

from sleep import Model


@pytest.fixture(name="diary")
def fixture_diary():
    return pd.read_csv("tutorial/diary.csv")


def test_unfied(diary):
    """
    Test the Model.compute method by verifying that the numerical solution matches the expected
    differential equations for homeostasis and debt:

    Differential equations:
        During wake:
            ds/dt = (1 - s) / t_w
            d(d)/dt = (-d + 1) / t_la
        During sleep:
            ds/dt = -(s - d) / t_s
            d(d)/dt = (-d - wsr) / t_la
        where:
            - s: homeostasis
            - d: debt
            - t_w: time constant during wake
            - t_s: time constant during sleep
            - t_la: time constant for debt
            - wsr: (T - need) / need (wake-sleep ratio)
            - T: period
            - need: sleep need
    """
    model = Model(name="unified", diary=diary)
    key = "Chronic Sleep Restriction: 21h / 3h → Recovery: 16h / 8h"
    res = 10  # min
    rtol = 0.1
    time = np.arange(0, 10 * 24 * 60, res)  # every 10 min
    time = time / 60  # min -> h
    out = model.compute(key, time)  # (1000, )
    s, d = out["homeostasis"], out["debt"]
    t_s = model.defaults["t_s"]
    t_w = model.defaults["t_w"]
    t_la = model.defaults["t_la"]
    T = model.defaults["T"]
    need = model.defaults["need"]
    wsr = (T - need) / need

    # Identify wake/sleep segments
    sleep_mask = np.zeros_like(time, dtype=bool)
    cd_key = diary["key"] == key
    for awake, asleep in zip(diary[cd_key]["awake"], diary[cd_key]["asleep"]):
        sleep_mask[(time >= asleep) & (time < awake)] = True
    wake_mask = ~sleep_mask

    # Compute ds/dt and d(d)/dt
    ds_dt = np.diff(s) / np.diff(time)
    dd_dt = np.diff(d) / np.diff(time)

    # During wake: ds/dt = (1-s)/t_w
    expected_ds_dt = (1 - s[wake_mask]) / t_w
    np.testing.assert_allclose(ds_dt[wake_mask[:-1]], expected_ds_dt, rtol=rtol)

    # During wake: d(d)/dt = (-d + 1) / t_la
    expected_dd_dt = (-d[wake_mask] + 1) / t_la
    np.testing.assert_allclose(dd_dt[wake_mask[:-1]], expected_dd_dt, rtol=rtol)

    # During sleep: ds/dt = -(s-d)/t_s
    expected_ds_dt = -(s[sleep_mask] - d[sleep_mask]) / t_s
    np.testing.assert_allclose(ds_dt[sleep_mask[:-1]], expected_ds_dt[:-1], rtol=rtol)

    # During sleep: d(d)/dt = (-d - wsr) / t_la
    expected_dd_dt = (-d[sleep_mask] - wsr) / t_la
    np.testing.assert_allclose(dd_dt[sleep_mask[:-1]], expected_dd_dt[:-1], rtol=rtol)
