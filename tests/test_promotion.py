import pytest
def decision(candidate_rmse, base_rmse, candidate_ms, base_ms):
    return candidate_rmse < .95*base_rmse and candidate_ms <= 2*base_ms

def test_promotes_when_both_conditions_pass():
    assert decision(90,100,10,10)

def test_rejects_when_rmse_fails():
    assert not decision(96,100,10,10)

def test_rejects_when_latency_fails():
    assert not decision(90,100,21,10)

def test_rejects_when_both_fail():
    assert not decision(100,100,30,10)

def test_exact_rmse_boundary_does_not_pass():
    assert not decision(95,100,10,10)
