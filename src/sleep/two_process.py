import numpy as np
import pandas as pd
import torch


class TwoProcess:
    """Original two_process sleep model computing homeostasis from Borbely et al.

    This model computes the sleep homeostasis pressure (s)
    based on sleep/wake transitions and physiological parameters. The computation
    is done using PyTorch for differentiability and GPU support.

    The model takes state changes (awake/asleep transitions) and computes the
    evolution of s over time using piece-wise analytic solutions for
    wake and sleep periods.

    Differential equations:
        During wake:
            ds/dt = (1 - s) / t_w
            
        During sleep:
            ds/dt = -s / t_s
            
        where:
            - s: homeostasis
            - t_w: time constant during wake
            - t_s: time constant during sleep
            

    References
    ----------
    P. Rajdev et al., "A unified mathematical model to quantify performance impairment
    for both chronic sleep restriction and total sleep deprivation.",
    from Journal of theoretical biology. 2013.
    """

    def __init__(self):
        self.outputs = ["homeostasis"]  

    def __str__(self) -> str:
        """Return a user-friendly string representation of the model."""
        return (
            "Two-process sleep model computing:\n"
            "- Homeostasis (s): sleep pressure\n"
            "During wake:\n"
            "  ds/dt = (1 - s) / t_w\n"
            "During sleep:\n"
            "  ds/dt = -s / t_s\n"
        )

    def infer_initial_state(self, sc, rested, t_s, t_w, need,
        T):
        """
        Walk *backwards* through the diary until the very first timestamp,
        starting from the fully-rested (s_w, d_w) at wake - sc[rested].
        """
        
        s1 = self._periodic_values(t_s, t_w, need, T)
        
        wake = False  # the first backstep segment is always sleep
        if rested > 0:
            for t0, t1 in zip(sc[:rested].flip(0), sc[1 : rested + 1].flip(0)):  # reverse iterate
                dt = (t1 - t0).reshape(1, 1)  # keep tensor shape
                # Ensure s1 stays within [0, 1]:
                if s1 < 0.05:
                    s1 = 0.05
                elif s1 > 0.9:
                    s1 = 0.9    
                
                elif wake:
                        s1 = self._wake_backstep(s1, dt, t_w)
                else:
                        s1 = self._sleep_backstep(s1, dt, t_s)
                
                wake = not wake

        return s1  # these are s0, d0 at the very first diary row

    def compute(
        self,
        sc: torch.Tensor,
        rested: torch.Tensor,
        time: torch.Tensor,
        t_s: torch.Tensor,
        t_w: torch.Tensor,
        need: torch.Tensor,
        T: torch.Tensor,
        
    ) -> dict[str, torch.Tensor]:
        """Compute homeostasis and debt."""
        pi = 1 #vérifier si ce paramètre est utile
        mi = time.numel()
        # ----- initialisations -----
        wake = True
        s0 = self.infer_initial_state(sc, rested, t_s, t_w,  need, T)
        
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
                s_seg = self._wake_step(s0, dt, t_w)
            else:  # sleep
                s_seg = self._sleep_step(s0, dt, t_s)
            if dt.numel() > 1 or last_seg:
                s[in_seg] = s_seg[:-1] if not last_seg else s_seg
            done += in_seg.sum()

            s0 = s_seg[-1:]  # (1, pi)
            wake = not wake

        # Dans le two_process model, la dette est la même chose que la homeostasie
        #d=s.clone() 

        return {"homeostasis": s.T}  # (m, pi)

    def check_params(
        self,
        t_s: torch.Tensor,
        t_w: torch.Tensor,
        need: torch.Tensor,
        T: torch.Tensor,
        
    ):
        """Validate parameters."""
        # Validate parameters
        if (t_s <= 0).any():
            raise ValueError("t_s must be positive")
        if (t_w <= 0).any():
            raise ValueError("t_w must be positive")
        if (T <= 0).any():
            raise ValueError("t_w must be positive")
        if (need <= 0).any():
            raise ValueError("t_w must be positive")
        

    # ──────────────────────────────────────────────────────────────
    #  Core maths - all torch, differentiable
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _periodic_values(t_s, t_w,  need, T):
        """Compute the periodic values for homeostasis."""
        wake_time =  T - need
        a = torch.exp(-wake_time / t_w)
        b = torch.exp(-need / t_s)

        s_w = (b * (1 - a)) / (1 - a * b) 
        print("periodic solution s_w:", s_w)
        return s_w

    @staticmethod
    def _wake_step(s0, dt, t_w):
        """Compute the wake time point from start condition."""
        s = 1.0 - (1.0 - s0) * torch.exp(-dt / t_w)  #  (mseg, pi)
        
        return s

    @staticmethod
    def _sleep_step(s0, dt, t_s):
        """Compute the sleep time point from start condition."""
        s = s0 * torch.exp(-dt / t_s)
        return s

    @staticmethod
    def _wake_backstep(s1, dt, t_w):
        """Compute the wake time point from end condition."""
        s0 = 1.0 - (1.0 - s1) * torch.exp(+dt / t_w)
        return s0

    @staticmethod
    def _sleep_backstep(s1, dt, t_s):
        """Inverse of the sleep time point from end condition."""
        s0 = s1 * torch.exp(dt / t_s)
        return s0

    def plot(
        self, ax, diary: pd.DataFrame, time: np.ndarray, homeostasis: np.ndarray
    ):
        """Plot homeostasis, debt, and sleep periods for a given key."""
        # Plot homeostasis and debt
        time = time / 24
        offset = time[0]  # offset to start at 0
        time -= offset
        #ax.plot(time, homeostasis, "orange", linestyle="dotted", label="Homeostasis")
        ax.plot(time, homeostasis, "green", label="Homeostasis")
        ax.set_xlim(time[0], time[-1])
        ax.grid(True, alpha=0.3)

        for i, (asleep, awake) in enumerate(zip(diary["asleep"], diary["awake"])):
            t0 = asleep / 24 - offset
            t1 = awake / 24 - offset
            ax.axvspan(t0, t1, color="gray", alpha=0.3, label="Sleep" if i == 0 else "")
