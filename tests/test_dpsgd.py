import numpy as np
import pytest

import gdpnum
from scipy.stats import norm


grid_step = 1e-3
noise_multiplier = 9.4
sample_rate = 2**14 / 50_000
num_steps = 2_000

real_mu = 1.567013
regret_bound = 1e-2


@pytest.fixture
def accountant():
    yield gdpnum.dpsgd.CTDAccountant


def test_get_mu_and_regret_for_dpsgd():
    mu, regret = gdpnum.dpsgd.get_mu_and_regret_for_dpsgd(
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        num_steps=num_steps,
        grid_step=grid_step,
    )

    assert mu == pytest.approx(real_mu, abs=1e-2)
    assert regret <= regret_bound


def test_accountant():
    acct = gdpnum.dpsgd.CTDAccountant()
    for i in range(num_steps):
        acct.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

    mu, regret = acct.get_mu_and_regret(grid_step=grid_step)

    assert mu == pytest.approx(real_mu, abs=1e-2)
    assert regret <= regret_bound

    assert len(acct.get_beta()) == 8
