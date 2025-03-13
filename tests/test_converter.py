import pytest

from gdpnum import PLDConverter

from dp_accounting.pld.privacy_loss_distribution import (
    from_gaussian_mechanism,
    from_laplace_mechanism,
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


@pytest.mark.parametrize("b", [1.25, 1.5, 2])
def test_laplace_has_nonzero_regret(b):
    pld = from_laplace_mechanism(b)
    converter = PLDConverter(pld)
    mu, regret = converter.get_mu_and_regret()
    assert regret > 1e-2
