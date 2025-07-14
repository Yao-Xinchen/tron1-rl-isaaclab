# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class RslRlPpoAlgorithmMlpCfg(RslRlPpoAlgorithmCfg):
    """Configuration of the runner for on-policy algorithms."""

    # runner_type: str = "OnPolicyRunner"

    obs_history_len: int = 1


@configclass
class EncoderCfg:
    output_detach : bool = True
    num_input_dim : int = MISSING
    num_output_dim : int = 3
    hidden_dims : list[int] = [256, 128]
    activation : str = "elu"
    orthogonal_init : bool = False


import os
import copy
import torch
def export_mlp_as_onnx(mlp, path, name, input_dim):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name + ".onnx")
    model = copy.deepcopy(mlp).to("cpu")
    model.eval()

    dummy_input = torch.randn(input_dim)
    input_names = ["mlp_input"]
    output_names = ["mlp_output"]

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported policy as onnx script to: ", path)

def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
