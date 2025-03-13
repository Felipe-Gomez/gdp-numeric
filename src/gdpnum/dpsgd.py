import numpy as np
from typing import Union
from dp_accounting.pld.privacy_loss_distribution import from_gaussian_mechanism

from .converter import PLDConverter


def get_mu_and_regret_for_dpsgd(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    grid_step=1e-4,
    err=1e-10,
):
    """
    Get GDP mu and regret for DP-SGD.
    """
    pld = from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=grid_step,
    ).self_compose(num_steps)
    converter = PLDConverter(pld)
    return converter.get_mu_and_regret(err)


class CTDAccountant:
    """
    Opacus-compatible Connect the Dots accountant which can return numeric GDP parameter.
    """

    def __init__(self):
        self.history = []

    def step(self, *, noise_multiplier, sample_rate):
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))

        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def get_pld(self, grid_step=1e-4, use_connect_dots=True):
        noise_multiplier, sample_rate, num_steps = self.history[0]
        pld = from_gaussian_mechanism(
            standard_deviation=noise_multiplier,
            sampling_prob=sample_rate,
            use_connect_dots=use_connect_dots,
            value_discretization_interval=grid_step,
        ).self_compose(num_steps)

        for noise_multiplier, sample_rate, num_steps in self.history[1:]:
            pld_new = from_gaussian_mechanism(
                standard_deviation=noise_multiplier,
                sampling_prob=sample_rate,
                use_connect_dots=use_connect_dots,
                value_discretization_interval=grid_step,
            ).self_compose(num_steps)
            pld = pld.compose(pld_new)

        return pld

    def get_epsilon(self, *, delta, **kwargs):
        pld = self.get_pld(**kwargs)
        return pld.get_epsilon_for_delta(delta)

    def get_beta(self, *, alpha, **kwargs):
        pld = self.get_pld(**kwargs)
        converter = PLDConverter(pld)
        return converter.get_beta(alpha)

    def get_mu_and_regret(self, **kwargs):
        pld = self.get_pld(**kwargs)
        converter = PLDConverter(pld)
        return converter.get_mu_and_regret()

    def get_advantage(self, **kwargs):
        pld = self.get_pld(**kwargs)
        return conversions.get_delta_for_epsilon(0.0)

    def __len__(self):
        total = 0
        for _, _, steps in self.history:
            total += steps
        return total

    def mechanism(self):
        return "ctd"

    # The following methods are copied from https://opacus.ai/api/_modules/opacus/accountants/accountant.html#IAccountant
    # to avoid the direct dependence on the opacus package.
    def get_optimizer_hook_fn(self, sample_rate: float):
        """
        Returns a callback function which can be used to attach to DPOptimizer
        Args:
            sample_rate: Expected sampling rate used for accounting
        """

        def hook_fn(optim):
            # This works for Poisson for both single-node and distributed
            # The reason is that the sample rate is the same in both cases (but in
            # distributed mode, each node samples among a subset of the data)
            self.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate * optim.accumulated_iterations,
            )

        return hook_fn

    def state_dict(self, destination=None):
        """
        Returns a dictionary containing the state of the accountant.
        Args:
            destination: a mappable object to populate the current state_dict into.
                If this arg is None, an OrderedDict is created and populated.
                Default: None
        """
        if destination is None:
            destination = {}
        destination["history"] = deepcopy(self.history)
        destination["mechanism"] = self.__class__.mechanism
        return destination

    def load_state_dict(self, state_dict):
        """
        Validates the supplied state_dict and populates the current
        Privacy Accountant's state dict.

        Args:
            state_dict: state_dict to load.

        Raises:
            ValueError if supplied state_dict is invalid and cannot be loaded.
        """
        if state_dict is None or len(state_dict) == 0:
            raise ValueError(
                "state dict is either None or empty and hence cannot be loaded"
                " into Privacy Accountant."
            )
        if "history" not in state_dict.keys():
            raise ValueError(
                "state_dict does not have the key `history`."
                " Cannot be loaded into Privacy Accountant."
            )
        if "mechanism" not in state_dict.keys():
            raise ValueError(
                "state_dict does not have the key `mechanism`."
                " Cannot be loaded into Privacy Accountant."
            )
        if self.__class__.mechanism != state_dict["mechanism"]:
            raise ValueError(
                f"state_dict of {state_dict['mechanism']} cannot be loaded into "
                f" Privacy Accountant with mechanism {self.__class__.mechanism}"
            )
        self.history = state_dict["history"]
