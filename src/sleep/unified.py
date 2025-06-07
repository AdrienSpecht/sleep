import numpy as np
import pandas as pd
import torch


class Unified:
    """Unified sleep model computing homeostasis and sleep debt.

    This model computes the sleep homeostasis pressure (s) and sleep debt (d)
    based on sleep/wake transitions and physiological parameters. The computation
    is done using PyTorch for differentiability and GPU support.

    The model takes state changes (awake/asleep transitions) and computes the
    evolution of s and d over time using piece-wise analytic solutions for
    wake and sleep periods.

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

    References
    ----------
    P. Rajdev et al., "A unified mathematical model to quantify performance impairment
    for both chronic sleep restriction and total sleep deprivation.",
    from Journal of theoretical biology. 2013.
    """

    def __init__(self):
        self.outputs = ["homeostasis", "debt"]

    def __str__(self) -> str:
        """Return a user-friendly string representation of the model."""
        return (
            "Unified sleep model computing:\n"
            "- Homeostasis (s): sleep pressure\n"
            "- Debt (d): sleep debt\n\n"
            "During wake:\n"
            "  ds/dt = (1 - s) / t_w\n"
            "  dd/dt = (-d + 1) / t_la\n\n"
            "During sleep:\n"
            "  ds/dt = -(s - d) / t_s\n"
            "  dd/dt = (-d - wsr) / t_la"
        )

    def compute(
        self,
        sc: torch.Tensor,
        time: torch.Tensor,
        t_s: torch.Tensor,
        t_w: torch.Tensor,
        t_la: torch.Tensor,
        need: torch.Tensor,
        T: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute homeostasis and debt."""
        wsr = (T - need) / need
        pi, mi = wsr.numel(), time.numel()

        # ----- initial conditions -----
        wake = True
        s0, d0, _, _ = self._periodic_values(t_s, t_w, t_la, wsr, T)
        prev = sc[0]

        s = torch.full((mi, pi), fill_value=torch.nan, device=time.device, dtype=time.dtype)
        d = torch.full_like(s, fill_value=torch.nan)

        # ----- propagate over sleep/wake segments -----
        done = 0
        total = torch.sum(~torch.isnan(time)).item()
        for nxt in sc[1:]:
            if done == total:
                break  # early exit if all time points computed

            in_seg = (prev <= time) & (time < nxt)
            t = torch.cat([time[in_seg], nxt.unsqueeze(0)])
            dt = (t - prev).reshape(-1, 1)  # (mseg, 1)

            if wake:
                s_seg = 1.0 - (1.0 - s0) * torch.exp(-dt / t_w)  #  (mseg, pi)
                d_seg = 1.0 - (1.0 - d0) * torch.exp(-dt / t_la)
            else:  # sleep
                s_seg, d_seg = self._sleep_step(s0, d0, dt, t_s, t_la, wsr)

            if dt.numel() > 1:
                s[in_seg] = s_seg[:-1]
                d[in_seg] = d_seg[:-1]
                done += in_seg.sum()

            s0, d0 = s_seg[-1:], d_seg[-1:]  # (1, pi)
            wake = not wake
            prev = nxt

        return {"homeostasis": s.T, "debt": d.T}  # (m, pi)

    def check_params(
        self,
        t_s: torch.Tensor,
        t_w: torch.Tensor,
        t_la: torch.Tensor,
        need: torch.Tensor,
        T: torch.Tensor,
    ):
        """Validate parameters."""
        # Validate parameters
        if (t_s <= 0).any():
            raise ValueError("t_s must be positive")
        if (t_w <= 0).any():
            raise ValueError("t_w must be positive")
        if (t_la <= 0).any():
            raise ValueError("t_la must be positive")
        if (need <= 0).any():
            raise ValueError("need must be positive")
        if (T <= 0).any():
            raise ValueError("T must be positive")

        # Validate relationships
        if (t_s >= t_la).any():
            raise ValueError("t_s must be < t_la")
        if (t_w >= t_la).any():
            raise ValueError("t_w must be < t_la")

    # ──────────────────────────────────────────────────────────────
    #  Core maths - all torch, differentiable
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _sleep_step(s0, d0, dt, t_s, t_la, wsr):
        d = -wsr + (d0 + wsr) * torch.exp(-dt / t_la)
        e_s = torch.exp(-dt / t_s)
        e_la = torch.exp(-dt / t_la)
        B = t_la / (t_la - t_s)
        s = -wsr + (s0 + wsr) * e_s + B * (d0 + wsr) * (e_la - e_s)
        return s, d

    @staticmethod
    def _periodic_values(t_s, t_w, t_la, wsr, T):
        tot_w = T * wsr / (wsr + 1.0)
        tot_s = T - tot_w
        a = torch.exp(-tot_w / t_la)
        b = torch.exp(-tot_s / t_la)
        c = torch.exp(-tot_w / t_w)
        d = torch.exp(-tot_s / t_s)

        d_w = (b * (1 - a + wsr) - wsr) / (1 - a * b)
        d_s = 1 - a + a * d_w

        e = t_la / (t_la - t_s)
        s_w = (wsr * (d - 1) + d * (1 - c) + e * (b - d) * (1 - a + a * d_w + wsr)) / (1 - d * c)
        s_s = 1 - c + c * s_w
        return s_w, d_w, s_s, d_s

    def plot(
        self, ax, diary: pd.DataFrame, time: np.ndarray, homeostasis: np.ndarray, debt: np.ndarray
    ):
        """Plot homeostasis, debt, and sleep periods for a given key."""
        # Plot homeostasis and debt
        time = time / 24
        ax.plot(time, homeostasis, "orange", linestyle="dotted", label="Homeostasis")
        ax.plot(time, debt, "green", label="Debt")
        ax.set_xlim(time[0], time[-1])
        ax.grid(True, alpha=0.3)

        for i, (asleep, awake) in enumerate(zip(diary["asleep"], diary["awake"])):
            ax.axvspan(
                asleep / 24,
                awake / 24,
                color="gray",
                alpha=0.3,
                label="Sleep" if i == 0 else "",
            )
