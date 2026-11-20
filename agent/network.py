"""The shared per-slot MLP and the backpressure-baseline Q function on it.

    Q(s, a) = V(s) + (1 / L) sum_l dev_l(a) * A(s, l)   dev_l = 1[a_l != a_BP(s, l)], L live slots
    V(s)    = mean over live slots of v(s, l)

The deviation term is a MEAN over live slots, not a sum (changed 2026-11-20).
Summed, the bootstrap added the target network's A over every slot the
online network selected -- 60-110 of them per state at 6x12, each taken from
its upper tail -- so per-slot approximation error entered every target
multiplied by the number of slots, against a reward whose scale does not
grow with N. Measured on the summed form: Q(s, a) +470 and targets +2400
against rewards in [-13, +0.3] after 66 episodes, and A rising as one offset
on every slot. It also made A's scale, and so kappa's meaning, depend on N D:
every kappa value used before this change is in different units.

One MLP maps each slot's features to (v, A). V pools v over the live slots,
so it is a function of the whole state at any N; A(s, l) is what flipping
backpressure's bit on slot l is worth. ``deviate`` is the only place
backpressure and the learned residual meet:

    deviate_l = gain * A(s, l) > kappa

gain is 1 in normal use and 0 for the regression guard (no slot deviates,
so the action is backpressure's, exactly). See agent.ddqn for why this form.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import N_FEATURES


class LinkMLP(nn.Module):
    """F features -> (v, A), applied independently to every slot."""

    def __init__(self, hidden: tuple[int, ...], init_std: float) -> None:
        super().__init__()
        layers, width = [], N_FEATURES
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 2)
        nn.init.normal_(self.head.weight, 0.0, init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(feats))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over the trailing (N, D) dims where mask, per sample: (B,)."""
    m = mask.to(x.dtype)
    return (x * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1.0)


class ResidualQ(nn.Module):
    def __init__(self, hidden: tuple[int, ...], init_std: float, kappa: float) -> None:
        super().__init__()
        self.net = LinkMLP(hidden, init_std)
        self.kappa = float(kappa)
        self.gain = 1.0

    def value_advantage(self, feats: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """V (B,) and A (B, N, D); A is 0 on padding."""
        out = self.net(feats)
        v = masked_mean(out[..., 0], mask)
        adv = out[..., 1] * mask.to(out.dtype)
        return v, adv

    def deviate(self, adv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, N, D) bool: slots where the agent overrides backpressure."""
        return (self.gain * adv > self.kappa) & mask


def joint_q(v: torch.Tensor, adv: torch.Tensor, dev: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,) Q(s, a) = V + (sum of A over deviating live slots) / (live slots)."""
    d = (dev & mask).to(adv.dtype)
    return v + (adv * d).flatten(1).sum(1) / mask.to(adv.dtype).flatten(1).sum(1).clamp(min=1.0)
