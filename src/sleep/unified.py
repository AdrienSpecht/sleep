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

    def infer_initial_state(self, sc, rested, t_s, t_w, t_la, need, T):
        """
        Walk *backwards* through the diary until the very first timestamp,
        starting from the fully-rested (s_w, d_w) at wake - sc[rested].
        """
        wsr = (T - need) / need
        s1, d1, _, _ = self._periodic_values(t_s, t_w, t_la, wsr, T)
        gap = s1 - d1  # use the gap to find s0

        wake = False  # the first backstep segment is always sleep
        if rested > 0:
            for t0, t1 in zip(sc[:rested].flip(0), sc[1 : rested + 1].flip(0)):  # reverse iterate
                dt = (t1 - t0).reshape(1, 1)  # keep tensor shape
                if wake:
                    d1 = self._wake_backstep(d1, dt, t_la)
                else:
                    d1 = self._sleep_backstep(d1, dt, t_la, wsr)
                wake = not wake
        s1 = d1.clone() + gap

        return s1, d1  # these are s0, d0 at the very first diary row

    def compute(
        self,
        sc: torch.Tensor,
        rested: torch.Tensor,
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

        # ----- initialisations -----
        wake = True
        s0, d0 = self.infer_initial_state(sc, rested, t_s, t_w, t_la, need, T)

        s = torch.full((mi, pi), torch.nan, device=time.device, dtype=time.dtype)
        d = torch.full_like(s, torch.nan)

        # ----- propagate over sleep/wake segments -----
        done = 0
        total = torch.sum(~torch.isnan(time)).item()
        for t0, t1 in zip(sc[:-1], sc[1:]):
            if done == total:
                break  # early exit if all time points computed

            in_seg = (t0 <= time) & (time < t1)
            last_seg = torch.isinf(t1)
            if not last_seg:  # need t1 for the next segment
                t = torch.cat([time[in_seg], t1.unsqueeze(0)])
            else:
                t = time[in_seg]
            dt = (t - t0).reshape(-1, 1)  # (mseg, 1)

            if wake:
                s_seg, d_seg = self._wake_step(s0, d0, dt, t_w, t_la)
            else:  # sleep
                s_seg, d_seg = self._sleep_step(s0, d0, dt, t_s, t_la, wsr)

            if dt.numel() > 1 or last_seg:
                s[in_seg] = s_seg[:-1] if not last_seg else s_seg
                d[in_seg] = d_seg[:-1] if not last_seg else d_seg
            done += in_seg.sum()

            s0, d0 = s_seg[-1:], d_seg[-1:]  # (1, pi)
            wake = not wake

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

    @staticmethod
    def _wake_step(s0, d0, dt, t_w, t_la):
        """Compute the wake time point from start condition."""
        s = 1.0 - (1.0 - s0) * torch.exp(-dt / t_w)  #  (mseg, pi)
        d = 1.0 - (1.0 - d0) * torch.exp(-dt / t_la)
        return s, d

    @staticmethod
    def _sleep_step(s0, d0, dt, t_s, t_la, wsr):
        """Compute the sleep time point from start condition."""
        e_s = torch.exp(-dt / t_s)
        e_la = torch.exp(-dt / t_la)
        B = t_la / (t_la - t_s)
        d = -wsr + (d0 + wsr) * e_la
        s = -wsr + (s0 + wsr) * e_s + B * (d0 + wsr) * (e_la - e_s)
        return s, d

    @staticmethod
    def _wake_backstep(d1, dt, t_la):
        """Compute the wake time point from end condition."""
        d0 = 1.0 - (1.0 - d1) * torch.exp(+dt / t_la)
        return d0

    @staticmethod
    def _sleep_backstep(d1, dt, t_la, wsr):
        """Inverse of the sleep time point from end condition."""
        e_la = torch.exp(-dt / t_la)
        d0 = -wsr + (d1 + wsr) / e_la
        return d0

    def plot(
        self, ax, diary: pd.DataFrame, time: np.ndarray, homeostasis: np.ndarray, debt: np.ndarray
    ):
        """Plot homeostasis, debt, and sleep periods for a given key."""
        # Plot homeostasis and debt
        time = time / 24
        time -= time[0]  # offset to start at 0
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
