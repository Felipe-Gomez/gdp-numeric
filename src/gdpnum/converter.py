"""
This module provides functions that compute a tradeoff curve T(P,Q)
given a PLRVs instance (Probability Loss Random Variables). We rely
on the notation from:

  - Dong et al. (https://arxiv.org/pdf/1905.02383)
  - Kulynych et al. (https://arxiv.org/pdf/2407.02191)
  - Gomez et al.

This docstring closely follows that from https://github.com/Felipe-Gomez/riskcal/blob/main/riskcal/plrv.py.

Below, we summarize the main concepts and notation needed to
understand how these functions work.

----------------------------------------------------------------------
DOMINATING PAIRS AND TRADEOFF FUNCTIONS
----------------------------------------------------------------------

Given a mechanism M, we say (P, Q) is a discrete-valued dominating
pair if, for all 0 <= alpha <= 1,

    T(P,Q)(alpha) <= T(M(D), M(D'))(alpha),

where T denotes the tradeoff function. (In this code, T(P,Q) is often
called f.)

We define random variables X and Y via:

    Y = log[P(o) / Q(o)]   with o ~ P,
    X = log[P(o') / Q(o')] with o' ~ Q.

In cases where (P, Q) have disjoint support, X can have a point mass
at -∞, and Y can have a point mass at +∞. Their domains are:

    Domain_X = {-∞} ∪ {x_0, x_1, ..., x_{k-1}}
             = {x_{-1}} ∪ {x_0, ..., x_{k-1}},

    Domain_Y = {y_0, y_1, ..., y_{l-1}} ∪ {+∞}
             = {y_0, ..., y_{l-1}} ∪ {y_l}.

----------------------------------------------------------------------
GOOGLE DP_ACCOUNTING LIBRARY AND DISCRETIZATION
----------------------------------------------------------------------

In practice, all PLRVs passed to this module come from the
`dp_accounting` library developed by Google. That library discretizes
the privacy loss random variables so that

    x_i = Δ * (x_0 + i), 0 <= i <= k - 1
    y_j = Δ * (y_0 + j), 0 <= j <= l - 1,

where the scalar Δ is the discretization parameter.
As the functions in this module are scale invarient to Δ
(see, e.g. Algorithm 1 in Kulynych et al.), we let Δ=1 throughout
this module. As a result, the finite domain of X and Y takes on
equally spaced integer points, plus possible point masses at
-∞ (for X) and +∞ (for Y).

----------------------------------------------------------------------
PIECEWISE LINEARITY AND ALPHA_BAR
----------------------------------------------------------------------

We let f = T(P,Q). In some cases, f(alpha) != f^{-1}(alpha). This
discrepancy often arises under Poisson subsampling, where one order
corresponds to remove-adjacent datasets (D → D') and the other to
add-adjacent datasets (D' → D). When this occurs, we must apply a
symmetrization step to obtain the symmetric tradeoff curve that
matches the add/remove neighboring relation. This is precisely the
process described in Definition F.1 of Dong et al. (arXiv:1905.02383).

To carry out this symmetrization, we need to find the smallest value
of alpha for which -1 lies in the subdifferential of f. We denote
this value alpha_bar.

It can be shown that f is piecewise linear with breakpoints at
(Pr[X > x_i], Pr[Y <= x_i]) for -1 ≤ i ≤ k - 1. The linear segment
to the right of breakpoint i has slope -e^(x_i). At each breakpoint
i < k - 1, the subdifferential is the interval
[-e^(x_{i+1}), -e^(x_i)].

Let m be such that x_m = 0 (i.e., m = -x_0). Then -1 appears in
subdifferentials m and m - 1:

    [-e^(x_{m+1}), -e^(x_m)]
    [-e^(x_m), -e^(x_{m-1})]

Noting that alpha decreases as the index increases, the smallest
alpha at which -1 appears in the subdifferential corresponds to i = m.
This yields:

    alpha_bar = Pr[X > x_m] = Pr[X > 0],
    beta_bar = Pr[Y <= x_m] = Pr[Y <= 0].

Once alpha_bar is determined, implementing the symmetrization process
(Definition F.1 of Dong et al.) is straightforward.
"""

from scipy.stats import norm
from typing import Union, Callable
import numpy as np


def _ensure_array(x: Union[np.ndarray, int, float]) -> np.ndarray:
    is_scalar = isinstance(x, (int, float))
    if is_scalar:
        return np.asarray([x]), is_scalar
    return np.asarray(x), False


def gaussian_fdp(mu: float) -> Callable[[float], float]:
    """
    Returns tradeoff curve of a mu_GDP mechanism.

    Parameters:
        - mu: float that denotes the parameter in mu-GDP.

    Returns:
        - Lambda function beta(alpha) = Phi( Phi^{-1}(1 - alpha) - mu).
    """
    # return lambda alpha: norm.cdf( -norm.ppf(alpha) - mu)
    return lambda alpha: norm.cdf(norm.isf(alpha) - mu)


def get_worst_case_regret(
    alphas: np.ndarray, betas: np.ndarray, mu: float, tol: float = 1e-10
) -> float:
    """
    Let f denote the piecewise linear tradeoff curve with breakpoints
    (alphas[i], betas[i]).

    This function finds the smallest value of Delta such that
    f(x + Delta) - Delta<= f_mu(x) for all x in alphas, where mu is
    the parameter in mu-GDP.

    Parameters:
        - alphas: array of x breakpoints for the tradeoff curve.
        - betas:  array of y breakpoints for the tradeoff curve.
        - mu: float that denotes the parameter in mu-GDP.
        - tol: Tolerance for the binary search. Defaults to 1e-10.

    Returns:
        - Delta: The smallest value of Delta satisfying the condition.
    """

    f = lambda alpha: np.interp(alpha, alphas, betas)

    # Define the search range for Delta
    delta_low = 0.0
    delta_high = 1.0

    # Define Gaussian tradeoff function
    g = gaussian_fdp(mu)

    # Grid of probabilities
    x = alphas

    while delta_high - delta_low > tol:

        delta_mid = (delta_low + delta_high) / 2

        # Evaluate the condition for the current Delta
        shifted_x = x + delta_mid

        # Ensure shifted_x does not exceed 1 (clipping to range [0, 1])
        shifted_x = np.clip(shifted_x, 0, 1)

        if np.all(f(shifted_x) - delta_mid <= g(x)):
            # If condition is satisfied, search lower Delta
            delta_high = delta_mid
        else:
            # Otherwise, search higher Delta
            delta_low = delta_mid

    # We want an underestimate of regret, so we return delta_low
    return delta_low


def _get_plrv(pld):
    """
    Extract lower loss, infinity mass, and pmf from a
    PrivacyLossDistribution object.

    Parameters:
        - pld: A PrivacyLossDistribution object.

    Returns:
        - Tuple of:
            - lower_loss: Integer denoting start of PLD support
            - infinity_mass: float denoting mass at + infinity in PLD
            - pmf: Array of PLD PMF values
    """
    pld = pld.to_dense_pmf()
    pmf = pld._probs
    lower_loss = pld._lower_loss
    infinity_mass = pld._infinity_mass
    return lower_loss, infinity_mass, pmf


def _normalize_pmf(pmf, infinity_mass):
    """
    Normalize a probability mass function (PMF) to sum to 1 given
    mass at infinity. The PMF is assumed to be over a finite support,
    and the mass at infinity is not accounted for in the PMF.

    Parameters:
        - pmf: Array of PMF values
        - infinity_mass: float denoting mass at + infinity.

    Returns:
        - Normalized pmf such that sum(pmf) + infinity_mass = 1
    """

    return pmf * (1 - infinity_mass) / np.sum(pmf)


def _compute_breakpoints(pld: "dp_accounting.PrivacyLossDistribution"):
    """
    Given a PrivacyLossDistribution object from dp_accounting,
    extract the PLD pmf and compute the correspond breakpoints in
    the tradeoff function. Also computes alpha_bar and beta_bar, which
    are needed downstream for symmetrization of the tradeoff function.

    Parameters:
        - pld: A PrivacyLossDistribution object.

    Returns:
        - Tuple of:
            - alphas: Array of x breakpoints for the tradeoff curve.
            - betas:  Array of y breakpoints for the tradeoff curve.
            - alpha_bar: float Pr[X > 0]
            - beta_bar:  float Pr[Y <= 0]
    """
    # Extract PLRV data for remove-adjacent and add-adjacent cases
    y0, mass_inf_Y, pmf_Y = _get_plrv(pld._pmf_remove)
    z0, mass_inf_Z, pmf_Z = _get_plrv(pld._pmf_add)
    z_max = z0 + len(pmf_Z) - 1

    # Ensure PMFs are valid (correct numerical errors)
    pmf_Y = _normalize_pmf(np.maximum(pmf_Y, 0), mass_inf_Y)
    pmf_Z = _normalize_pmf(np.maximum(pmf_Z, 0), mass_inf_Z)

    # Transform from PLRV Z to PLRV X (X = -Z)
    pmf_X = pmf_Z[::-1]
    x0 = -z_max

    # Compute the max values in the support of X and Y
    k = len(pmf_X)
    l = len(pmf_Y)
    x_max = x0 + k - 1
    y_max = y0 + l - 1

    # Compute index mappings between X and Y domains
    y0_idx_X = y0 - x0
    x0_idx_Y = x0 - y0
    x_max_idx_Y = x0_idx_Y + k - 1

    # Compute start index for the X cumulative sums
    start_x = max(0, y0_idx_X)

    # Determine if x_max > y_max
    x_exceeds_y = x_max > y_max

    # Compute the alpha breakpoints
    cumsum_X = np.cumsum(pmf_X[start_x:][::-1])
    slice_idx = x_max - y_max - 1 if x_exceeds_y else 0
    alphas = np.hstack((0, cumsum_X[slice_idx:], 1))

    # Compute start index for the Y cumulative sums
    start_y = max(0, x0_idx_Y)

    # Compute beta breakpoints
    cumsum_Y = np.cumsum(pmf_Y[: x_max_idx_Y + 1])[start_y:][::-1]
    betas = np.hstack((cumsum_Y, 0, 0))

    # Adjust betas based on x_exceeds_y
    if x_exceeds_y:
        betas = np.hstack((betas[0], betas))

    # Compute alpha bar and beta_bar
    zero_index_X = -x0
    zero_index_Y = -y0
    alpha_bar = np.sum(pmf_X[zero_index_X + 1 :])
    beta_bar = np.sum(pmf_Y[: zero_index_Y + 1])
    return alphas, betas, alpha_bar, beta_bar


class PLDConverter:
    """
    Converts a PrivacyLossDistribution (PLD) object from the `dp_accounting`
    library into a piecewise linear tradeoff function representation.

    This class computes and validates the tradeoff function breakpoints
    (alphas, betas) derived from the PLD object and provides methods for:
        - Evaluating the tradeoff function and its inverse.
        - Computing a symmetric version of the tradeoff function.
        - Extracting a pessimistic value for μ (i.e. so that f_μ <= T(P,Q)),
          and worst-case regret.

    Attributes:
        alphas (np.ndarray): Breakpoints for the tradeoff function.
        betas (np.ndarray): Corresponding tradeoff values.
        alphas_symm (np.ndarray): Symmetrized breakpoints (if needed).
        betas_symm (np.ndarray): Symmetrized tradeoff values (if needed).
        is_symmetric (bool): Whether the PLD tradeoff function is symmetric.

    Methods:
        tradeoff_function(input_alphas):
            Computes the tradeoff function T(P, Q) for given input alpha values.
        inverse_tradeoff_function(input_alphas):
            Computes the inverse tradeoff function T(Q, P).
        get_beta(input_alphas):
            Computes symmetrized beta values corresponding to given input alpha values.
        get_mus(err=1e-10):
            Computes the privacy parameter μ using tradeoff curve properties.
        get_mu_and_regret(err=1e-10):
            Computes the pessimistic μ and worst-case regret.
    """

    def __init__(self, pld: "dp_accounting.PrivacyLossDistribution"):
        # Compute breakpoints and alpha_bar, beta_bar on tradeoff curve
        alphas, betas, alpha_bar, beta_bar = _compute_breakpoints(pld)

        # Validate monotonicity of breakpoints
        assert np.all(np.diff(alphas) >= 0)
        assert np.all(np.diff(betas[::-1]) >= 0)

        self.alphas = alphas
        self.betas = betas
        self.is_symmetric = pld._symmetric

        # If pld is symmetric, no need to symmetrize
        if self.is_symmetric:
            self.alphas_symm = self.alphas
            self.betas_symm = self.betas

        else:

            # Linear Interoplation
            if alpha_bar <= beta_bar:

                alpha_bar_index = np.searchsorted(alphas, alpha_bar)
                self.alphas_symm, self.betas_symm = self._compute_symm_tradeoff_points(
                    alpha_bar_index
                )

            # Max
            else:
                self.alphas_symm, self.betas_symm = self._compute_max_tradeoff_points()

    def _compute_max_tradeoff_points(self):
        # Combine all unique breakpoints from x and y
        combined_points = np.unique(np.concatenate([self.alphas, self.betas]))

        # Evaluate f(x) and f_inverse(x) at these combined points
        f_values = np.interp(combined_points, self.alphas, self.betas)
        f_inverse_values = np.interp(
            combined_points, self.betas[::-1], self.alphas[::-1]
        )

        # Take the maximum at each combined point
        max_values = np.maximum(f_values, f_inverse_values)

        return combined_points, max_values

    def _compute_symm_tradeoff_points(self, alpha_bar_index):

        alphas_symm = np.hstack(
            (
                self.alphas[: alpha_bar_index + 1],
                self.betas[: alpha_bar_index + 1][::-1],
            )
        )
        betas_symm = np.hstack(
            (
                self.betas[: alpha_bar_index + 1],
                self.alphas[: alpha_bar_index + 1][::-1],
            )
        )

        assert np.all(np.diff(alphas_symm) >= 0)
        return alphas_symm, betas_symm

    def tradeoff_function(self, input_alphas: np.ndarray) -> np.ndarray:
        """Evaluate the tradeoff function at given points."""
        return np.interp(input_alphas, self.alphas, self.betas)

    def inverse_tradeoff_function(self, input_alphas) -> np.ndarray:
        """Evaluate the inverse tradeoff function at given points."""
        return np.interp(input_alphas, self.betas[::-1], self.alphas[::-1])

    def get_beta(
        self, input_alphas: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Evaluate the symmetrized tradeoff function at given points."""

        # Convert alphas to array; check if input was scalar
        input_alphas, is_scalar = _ensure_array(input_alphas)

        # Symmetric implies T(P,Q) = T(Q,P). No need to symmetrize
        if self.is_symmetric:
            output = self.tradeoff_function(input_alphas)

        else:
            output = np.interp(input_alphas, self.alphas_symm, self.betas_symm)

        if is_scalar:
            return output.item()

        return output

    def _get_mus(self, err=1e-10):
        alphas = self.alphas_symm
        betas = self.betas_symm

        valid_alphas = (alphas > err) & (alphas < 1 - err)
        valid_betas = (betas > err) & (betas < 1 - err)
        valid_curve = betas <= 1 - alphas - err  # makes sure beta < 1 - alpha

        valid_mask = valid_alphas & valid_betas & valid_curve

        alphas = alphas[valid_mask]
        betas = betas[valid_mask]

        mus = norm.isf(alphas) + norm.isf(betas)

        return np.min(mus), np.max(mus)

    def get_mu(self, err: float = 1e-10) -> float:
        _, pess_mu = self._get_mus(err)
        return pess_mu

    def get_mu_and_regret(self, err: float = 1e-10) -> tuple[float, float]:
        _, pess_mu = self._get_mus(err)

        # define alpha grid and get finite points on symmtric tradeoff curve
        alphas = self.alphas_symm
        betas = self.betas_symm

        # get worst case regret
        regret = get_worst_case_regret(alphas, betas, pess_mu, tol=err)

        return pess_mu, regret
