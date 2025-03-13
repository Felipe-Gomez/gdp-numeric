# gdpnum

[![CI](https://github.com/Felipe-Gomez/gdp-numeric/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Felipe-Gomez/gdp-numeric/actions/workflows/ci.yml)

Repository for numerically computing the privacy parameter in Gaussian Differential Privacy.

## Quickstart

The library relies on the privacy loss distribution accounting from the `dp_accounting` package.
Here is an example of using it with DP-SGD:

```python
from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism

noise_multiplier = 9.4
sample_rate = 0.328
steps = 2_000

# Construct the PLD object.
pld = from_gaussian_mechanism(
    standard_deviation=sigma,
    sampling_prob=sample_rate,
    use_connect_dots=True,
    value_discretization_interval=1e-3,
).self_compose(steps)
```

Using the `PLDConverter` object, we can numerically compute the GDP mu parameter, and regret which
shows the goodness-of-fit of the GDP:

```python
import gdpnum

converter = gdpnum.PLDConverter(pld)
mu, regret = converter.get_mu_and_regret()
# (1.5685621993129137, 0.0010208130697719753)
```
