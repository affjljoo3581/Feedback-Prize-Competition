from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

try:
    from apex.normalization import FusedLayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm as FusedLayerNorm


def get_parameter_groups(module: nn.Module) -> List[Dict[str, Any]]:
    """Get parameter groups for transformer training.

    It is well-known that excluding layer-norm and bias parameters from weight-decay
    leads better performance at training transformer-based models. To achieve that, this
    function creates the separated parameter groups for applying weight-decay and
    ignoring weight-decay.

    Args:
        module: The target module to get the parameters from.

    Returns:
        A list of two parameter groups.
    """
    do_decay = [p for p in module.parameters() if p.ndim < 2]
    no_decay = [p for p in module.parameters() if p.ndim >= 2]
    return [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]


def replace_with_fused_layernorm(module: nn.Module):
    """Replace the normal (PyTorch-vanilla) layer-norms to apex fused layer-norms.

    Args:
        module: The target module to be replaced.
    """
    for submodule in module.modules():
        for name, layer in submodule.named_children():
            if not isinstance(layer, nn.LayerNorm):
                continue

            # Create new fused layer-norm and copy the original parameters.
            new_layer = FusedLayerNorm(layer.normalized_shape, layer.eps)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias

            # Replace the layer-norm to the new one.
            setattr(submodule, name, new_layer)


def reinit_last_layers(model: PreTrainedModel, num_layers: int):
    """Re-initialize the last-k transformer layers.

    Args:
        model: The target transformer model.
        num_layers: The number of layers to be re-initialized.
    """
    if num_layers > 0:
        base_model = getattr(model, model.base_model_prefix)
        base_model.encoder.layer[-num_layers:].apply(model._init_weights)


def concat_tensors_with_padding(
    tensor_list: List[torch.Tensor], padding: Union[int, float] = 0
) -> torch.Tensor:
    """Concatenate the list of tensors to be a single tensor with paddings.

    Args:
        tensor_list: The list of tensors which have different lengths. They should have
            the shape of `(batch_size, seq_len, dim)` or `(batch_size, seq_len)`.
        padding: The padding value for the tensors. If the tensor is shorter than other
            tensors, than it will be padded with this value. Default is `0`.

    Returns:
        A concatenated single tnesor.
    """
    max_length = max(x.size(1) for x in tensor_list)

    padded_tensor_list = []
    for tensor in tensor_list:
        # This function only supports two and three dimensional tensors.
        if tensor.ndim == 2:
            padding_size = (0, max_length - tensor.size(1))
        elif tensor.ndim == 3:
            padding_size = (0, 0, 0, max_length - tensor.size(1))

        padded_tensor_list.append(F.pad(tensor, padding_size, value=padding))
    return torch.cat(padded_tensor_list)
