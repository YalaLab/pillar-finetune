# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
from pillar.utils.logging import logger


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    config = model.backbone_model.visual.model_config

    num_layers = sum(model.stages()) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, model.stages())
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_vit(name, stages):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    name = name.replace("backbone_model.model.model.", "").replace("visual.", "")
    if any([layer in name for layer in ["cls_token", "pos_embed", "logit_scale", "patch_embed"]]):
        return 0
    elif name.startswith("atlas_models."):
        name = name.strip("atlas_models.")

        layers = [int(l) for l in name.split(".") if l.isdigit()]
        current_layer = 0
        if layers[0] == 0:
            return layers[1]
        elif layers[0] == 1:
            return stages[0] + layers[1]
        else:
            return stages[0] + stages[1] + layers[1]
    else:
        return sum(stages) + 1
