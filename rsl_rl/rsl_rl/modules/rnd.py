# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128], activation='elu'):
        super(MLP, self).__init__()
        activation = get_activation(activation)

        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
        encoder_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], output_dim))
                torch.nn.init.orthogonal_(encoder_layers[-1].weight, 0.01)
                torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
                torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        latent = nn.functional.normalize(latent, p=2, dim=-1)
        return latent


class RandomNetworkDistillation(nn.Module):
    """Implementation of Random Network Distillation (RND) [1]

    References:
        .. [1] Burda, Yuri, et al. "Exploration by random network distillation." arXiv preprint arXiv:1810.12894 (2018).
    """

    def __init__(
        self,
        num_states: int,
        num_outputs: int,
        predictor_hidden_dims: list[int],
        target_hidden_dims: list[int],
        activation: str = "elu",
        weight: float = 0.0,
        state_normalization: bool = False,
        reward_normalization: bool = False,
        device: str = "cpu",
        weight_schedule: dict | None = None,
    ):
        """Initialize the RND module.

        - If :attr:`state_normalization` is True, then the input state is normalized using an Empirical Normalization layer.
        - If :attr:`reward_normalization` is True, then the intrinsic reward is normalized using an Empirical Discounted
          Variation Normalization layer.

        .. note::
            If the hidden dimensions are -1 in the predictor and target networks configuration, then the number of states
            is used as the hidden dimension.

        Args:
            num_states: Number of states/inputs to the predictor and target networks.
            num_outputs: Number of outputs (embedding size) of the predictor and target networks.
            predictor_hidden_dims: List of hidden dimensions of the predictor network.
            target_hidden_dims: List of hidden dimensions of the target network.
            activation: Activation function. Defaults to "elu".
            weight: Scaling factor of the intrinsic reward. Defaults to 0.0.
            state_normalization: Whether to normalize the input state. Defaults to False.
            reward_normalization: Whether to normalize the intrinsic reward. Defaults to False.
            device: Device to use. Defaults to "cpu".
            weight_schedule: The type of schedule to use for the RND weight parameter.
                Defaults to None, in which case the weight parameter is constant.
                It is a dictionary with the following keys:

                - "mode": The type of schedule to use for the RND weight parameter.
                    - "constant": Constant weight schedule.
                    - "step": Step weight schedule.
                    - "linear": Linear weight schedule.

                For the "step" weight schedule, the following parameters are required:

                - "final_step": The step at which the weight parameter is set to the final value.
                - "final_value": The final value of the weight parameter.

                For the "linear" weight schedule, the following parameters are required:
                - "initial_step": The step at which the weight parameter is set to the initial value.
                - "final_step": The step at which the weight parameter is set to the final value.
                - "final_value": The final value of the weight parameter.
        """
        # initialize parent class
        super().__init__()

        # Store parameters
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.initial_weight = weight
        self.device = device
        self.state_normalization = state_normalization
        self.reward_normalization = reward_normalization

        # Normalization disabled (not available in this version)
        self.state_normalizer = torch.nn.Identity()
        self.reward_normalizer = torch.nn.Identity()

        # counter for the number of updates
        self.update_counter = 0

        # resolve weight schedule
        if weight_schedule is not None:
            self.weight_scheduler_params = weight_schedule
            self.weight_scheduler = getattr(self, f"_{weight_schedule['mode']}_weight_schedule")
        else:
            self.weight_scheduler = None
        # Create network architecture
        self.predictor = MLP(num_states, num_outputs, predictor_hidden_dims, activation).to(self.device)
        self.target = MLP(num_states, num_outputs, target_hidden_dims, activation).to(self.device)

        # make target network not trainable
        self.target.eval()

    def get_intrinsic_reward(self, state) -> torch.Tensor:
        # Note: the counter is updated number of env steps per learning iteration
        self.update_counter += 1
        # Normalize the state
        rnd_state = self.state_normalizer(state)
        # Obtain the embedding of the rnd state from the target and predictor networks
        target_embedding = self.target(rnd_state).detach()
        predictor_embedding = self.predictor(rnd_state).detach()
        # Compute the intrinsic reward as the distance between the embeddings
        intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)
        # Normalize intrinsic reward
        intrinsic_reward = self.reward_normalizer(intrinsic_reward)

        # Check the weight schedule
        if self.weight_scheduler is not None:
            self.weight = self.weight_scheduler(step=self.update_counter, **self.weight_scheduler_params)
        else:
            self.weight = self.initial_weight
        # Scale intrinsic reward
        intrinsic_reward *= self.weight

        return intrinsic_reward

    def forward(self, *args, **kwargs):
        raise RuntimeError("Forward method is not implemented. Use get_intrinsic_reward instead.")

    def train(self, mode: bool = True):
        # sets module into training mode
        self.predictor.train(mode)
        return self

    def eval(self):
        return self.train(False)

    """
    Different weight schedules.
    """

    def _constant_weight_schedule(self, step: int, **kwargs):
        return self.initial_weight

    def _step_weight_schedule(self, step: int, final_step: int, final_value: float, **kwargs):
        return self.initial_weight if step < final_step else final_value

    def _linear_weight_schedule(self, step: int, initial_step: int, final_step: int, final_value: float, **kwargs):
        if step < initial_step:
            return self.initial_weight
        elif step > final_step:
            return final_value
        else:
            return self.initial_weight + (final_value - self.initial_weight) * (step - initial_step) / (
                final_step - initial_step
            )


def resolve_rnd_config(alg_cfg, state_dim, env):
    """Resolve the RND configuration.

    Args:
        alg_cfg: The algorithm configuration dictionary.
        state_dim: The dimension of the state to use for RND.
        env: The environment.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # resolve dimension of rnd state
    if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
        # add rnd state dimension to config
        alg_cfg["rnd_cfg"]["num_states"] = state_dim
        # scale down the rnd weight with timestep
        alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt
    return alg_cfg


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
