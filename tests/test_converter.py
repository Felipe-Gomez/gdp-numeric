import numpy as np
import pytest

from scipy import stats

from gdpnum import PLDConverter

from dp_accounting.pld.privacy_loss_distribution import (
    from_gaussian_mechanism,
    from_randomized_response,
)

TOL = 1e-4


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0, 2.5])
def test_gaussian_converter_has_no_regret(mu):
    pld = from_gaussian_mechanism(1 / mu)
    converter = PLDConverter(pld)

    assert mu == pytest.approx(converter.get_mu(), abs=TOL)
    mu_prime, regret = converter.get_mu_and_regret()
    assert mu == pytest.approx(mu_prime, abs=TOL)
    assert regret < TOL


@pytest.mark.parametrize("eps", [0.75, 1.0, 1.5, 2.0])
def test_rr_has_nonzero_regret(eps):
    p = 2 / (1 + np.exp(eps))
    pld = from_randomized_response(p, num_buckets=2)
    converter = PLDConverter(pld)
    mu, regret = converter.get_mu_and_regret()

    # Proposition 6.1 in the paper.
    assert mu == pytest.approx(-2 * stats.norm.ppf(1 / (np.exp(eps) + 1)))
    assert regret > 0.01
