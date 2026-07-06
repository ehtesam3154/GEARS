"""Tests for the ported high-noise score-loss terms (HI_BOOST + FX_HI).

Run: PYTHONPATH=<repo> python gears/training/test_score_hi_boost.py
"""
import torch

from gears.training.score_hi_boost import (
    HiBoostState, high_noise_gate, hi_boost_multiplier, fx_hi_loss)
from gears.training.losses_geom import edm_residual_score_loss, AdaptiveQuantileGate

torch.manual_seed(0)
B, N, D = 8, 64, 16
SDR = 0.033
mask = torch.ones(B, N)
sigma = torch.linspace(0.01, 0.6, B)          # low -> high noise
R_target = torch.randn(B, N, D)
R_t = R_target + sigma.view(-1, 1, 1) * torch.randn(B, N, D)


def test_fx_hi_perfect_prediction_is_zero():
    gate = torch.ones(B)
    w = torch.ones(B)
    L, ratio = fx_hi_loss(R_target.clone(), R_target, R_t, sigma, SDR, mask, gate, w)
    assert L.item() < 1e-6, f"perfect F_x prediction should give ~0 loss, got {L.item()}"
    assert ratio is None or abs(ratio - 1.0) < 1e-3, f"F_x ratio should be ~1, got {ratio}"
    print(f"  [ok] fx_hi perfect: L={L.item():.2e} ratio={ratio}")


def test_fx_hi_imperfect_is_positive():
    gate = torch.ones(B)
    w = torch.ones(B)
    R0 = R_target + 0.3 * torch.randn(B, N, D)
    L, ratio = fx_hi_loss(R0, R_target, R_t, sigma, SDR, mask, gate, w)
    assert L.item() > 0, "imperfect F_x prediction should give positive loss"
    print(f"  [ok] fx_hi imperfect: L={L.item():.4f} ratio={ratio}")


def test_hi_boost_not_ready_is_identity():
    st = HiBoostState()                       # ready=False, last_fx_ratio_hi=None
    gate_hi = torch.ones(B)
    boost = hi_boost_multiplier(gate_hi, sigma, st, global_step=10)
    assert torch.allclose(boost, torch.ones(B)), "not-ready boost must be identity"
    print("  [ok] hi_boost not-ready -> identity")


def test_hi_boost_readiness_then_ramp():
    st = HiBoostState(min_stable_checks=3)
    gate_hi = torch.ones(B)
    # feed a stable fx ratio (=1.0 -> |log|=0 < tol) enough times to become ready
    for step in range(5):
        st.last_fx_ratio_hi = 1.0
        boost = hi_boost_multiplier(gate_hi, sigma, st, global_step=step)
    assert st.ready, "should be ready after >=min_stable_checks stable ratios"
    # once ready, at start_step ramp=0 -> boost ~1; later steps ramp up toward 4x
    st2 = HiBoostState(ready=True, start_step=0)
    b0 = hi_boost_multiplier(torch.ones(B), sigma, st2, global_step=0, boost_factor=4.0, ramp_steps=200)
    b_mid = hi_boost_multiplier(torch.ones(B), sigma, st2, global_step=100, boost_factor=4.0, ramp_steps=200)
    b_full = hi_boost_multiplier(torch.ones(B), sigma, st2, global_step=200, boost_factor=4.0, ramp_steps=200)
    assert b0.mean() < b_mid.mean() < b_full.mean(), "boost should ramp up over steps"
    # non-tail full-ramp boost approaches 4x (tail is capped at 2x)
    assert b_full.max() <= 4.0 + 1e-5 and b_full.max() > 3.0, f"full boost ~4x, got {b_full.max()}"
    print(f"  [ok] hi_boost ramp: b0={b0.mean():.2f} mid={b_mid.mean():.2f} full={b_full.mean():.2f} max={b_full.max():.2f}")


def test_hi_boost_tail_capped():
    st = HiBoostState(ready=True, start_step=0)
    gate_hi = torch.ones(B)
    boost = hi_boost_multiplier(gate_hi, sigma, st, global_step=200, boost_factor=4.0,
                                ramp_steps=200, tail_quantile=0.95, tail_cap=2.0)
    # the single tail sample (highest sigma) is capped at 2x while others reach 4x
    assert boost[-1].item() <= 2.0 + 1e-5, f"tail sample must be capped at 2x, got {boost[-1].item()}"
    print(f"  [ok] tail cap: tail_boost={boost[-1].item():.2f} (<=2.0), body_max={boost[:-1].max():.2f}")


def test_score_loss_flags_off_unchanged():
    R0 = R_target + 0.2 * torch.randn(B, N, D)
    base = edm_residual_score_loss(R0, R_target, sigma, SDR, mask)
    off = edm_residual_score_loss(R0, R_target, sigma, SDR, mask,
                                  R_t=R_t, hi_boost=False, fx_hi=False)
    assert torch.allclose(base, off), "flags-off must equal baseline"
    print(f"  [ok] flags off unchanged: {base.item():.4f}")


def test_score_loss_fx_hi_adds_term():
    R0 = R_target + 0.2 * torch.randn(B, N, D)
    gate = AdaptiveQuantileGate(target_rate=0.25, mode="high", warmup_steps=0, update_every=1)
    st = HiBoostState()
    base = edm_residual_score_loss(R0, R_target, sigma, SDR, mask)
    with_fx = edm_residual_score_loss(R0, R_target, sigma, SDR, mask, R_t=R_t,
                                      score_hi_gate=gate, boost_state=st, fx_hi=True,
                                      fx_hi_weight=2.0, global_step=5)
    assert with_fx.item() >= base.item(), "fx_hi should add a non-negative term"
    assert torch.isfinite(with_fx), "fx_hi loss must be finite"
    assert st.last_fx_ratio_hi is not None, "fx_hi should refresh boost readiness ratio"
    print(f"  [ok] fx_hi adds term: base={base.item():.4f} with_fx={with_fx.item():.4f} ratio={st.last_fx_ratio_hi:.3f}")


def test_score_loss_hi_boost_reweights_when_ready():
    R0 = R_target + 0.2 * torch.randn(B, N, D)
    gate = AdaptiveQuantileGate(target_rate=0.25, mode="high", warmup_steps=0, update_every=1)
    # prime the gate so it has a threshold, then force readiness
    for _ in range(5):
        high_noise_gate(sigma, SDR, gate)
    st = HiBoostState(ready=True, start_step=0)
    base = edm_residual_score_loss(R0, R_target, sigma, SDR, mask)
    boosted = edm_residual_score_loss(R0, R_target, sigma, SDR, mask, R_t=R_t,
                                      score_hi_gate=gate, boost_state=st, hi_boost=True,
                                      boost_factor=4.0, global_step=500)
    assert torch.isfinite(boosted), "boosted loss must be finite"
    assert not torch.allclose(base, boosted), "ready hi_boost should change the loss"
    print(f"  [ok] hi_boost reweights: base={base.item():.4f} boosted={boosted.item():.4f}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} score_hi_boost tests...")
    for t in tests:
        t()
    print("ALL PASSED")
