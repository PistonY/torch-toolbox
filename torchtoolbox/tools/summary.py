# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['summary']

from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np


def _flops_str(flops):
    preset = [(1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')]

    for p in preset:
        if flops // p[0] > 0:
            N = flops / p[0]
            ret = "%.1f%s" % (N, p[1])
            return ret
    ret = "%.1f" % flops
    return ret


def _cac_grad_params(p, w):
    t, n = 0, 0
    if w.requires_grad:
        t += p
    else:
        n += p
    return t, n


def _cac_msa(layer, input, output):
    sl, b, dim = output[0].size()
    assert b == 1, 'Only support batch size of 1.'
    tb_params = 0
    ntb__params = 0
    flops = 0

    if layer._qkv_same_embed_dim is False:
        tb_params += layer.q_proj_weight.numel()
        tb_params += layer.k_proj_weight.numel()
        tb_params += layer.v_proj_weight.numel()
    else:
        tb_params += layer.in_proj_weight.numel()

    if hasattr(layer, 'in_proj_bias'):
        tb_params += layer.in_proj_bias.numel()

    tb_params += layer.embed_dim**2

    # flops of this layer if fixed.
    # first get KQV
    flops += sl * dim * 3 * (2 * dim - 1)
    if hasattr(layer, 'in_proj_bias'):
        flops += dim * 3
    # then cac sa
    num_heads = layer.num_heads
    head_dim = layer.head_dim
    flops += (num_heads * sl * sl * (2 * head_dim - 1) + num_heads * sl * head_dim * (2 * sl - 1))
    # last linear
    flops += sl * dim * (2 * dim - 1) + dim
    return tb_params, ntb__params, flops


def _cac_conv(layer, input, output):
    # bs, ic, ih, iw = input[0].shape
    oh, ow = output.shape[-2:]
    kh, kw = layer.kernel_size
    ic, oc = layer.in_channels, layer.out_channels
    g = layer.groups

    tb_params = 0
    ntb__params = 0
    flops = 0
    if hasattr(layer, 'weight') and hasattr(layer.weight, 'shape'):
        params = np.prod(layer.weight.shape)
        t, n = _cac_grad_params(params, layer.weight)
        tb_params += t
        ntb__params += n
        flops += (2 * ic * kh * kw - 1) * oh * ow * (oc // g)
    if hasattr(layer, 'bias') and hasattr(layer.bias, 'shape'):
        params = np.prod(layer.bias.shape)
        t, n = _cac_grad_params(params, layer.bias)
        tb_params += t
        ntb__params += n
        flops += oh * ow * (oc // g)
    return tb_params, ntb__params, flops


def _cac_xx_norm(layer, input, output):
    tb_params = 0
    ntb__params = 0
    if hasattr(layer, 'weight') and hasattr(layer.weight, 'shape'):
        params = np.prod(layer.weight.shape)
        t, n = _cac_grad_params(params, layer.weight)
        tb_params += t
        ntb__params += n
    if hasattr(layer, 'bias') and hasattr(layer.bias, 'shape'):
        params = np.prod(layer.bias.shape)
        t, n = _cac_grad_params(params, layer.bias)
        tb_params += t
        ntb__params += n
    if hasattr(layer, 'running_mean') and hasattr(layer.running_mean, 'shape'):
        params = np.prod(layer.running_mean.shape)
        ntb__params += params
    if hasattr(layer, 'running_var') and hasattr(layer.running_var, 'shape'):
        params = np.prod(layer.running_var.shape)
        ntb__params += params
    in_shape = input[0]
    flops = np.prod(in_shape.shape)
    if layer.affine:
        flops *= 2
    return tb_params, ntb__params, flops


def _cac_linear(layer, input, output):
    ic, oc = layer.in_features, layer.out_features

    tb_params = 0
    ntb__params = 0
    flops = 0

    input = input[0]
    in_len = len(input.size())
    if in_len == 2:
        if hasattr(layer, 'weight') and hasattr(layer.weight, 'shape'):
            params = np.prod(layer.weight.shape)
            t, n = _cac_grad_params(params, layer.weight)
            tb_params += t
            ntb__params += n
            flops += (2 * ic - 1) * oc
        if hasattr(layer, 'bias') and hasattr(layer.bias, 'shape'):
            params = np.prod(layer.bias.shape)
            t, n = _cac_grad_params(params, layer.bias)
            tb_params += t
            ntb__params += n
            flops += oc
        return tb_params, ntb__params, flops
    elif in_len == 3:
        if input.size(0) == 1:
            sl, dim = input.shape[1:]
        elif input.size(1) == 1:
            sl, _, dim = input.shape
        else:
            raise ValueError('Only support batch size of 1.')

        if hasattr(layer, 'weight') and hasattr(layer.weight, 'shape'):
            params = np.prod(layer.weight.shape)
            t, n = _cac_grad_params(params, layer.weight)
            tb_params += t
            ntb__params += n
            flops += sl * (2 * ic - 1) * oc
        if hasattr(layer, 'bias') and hasattr(layer.bias, 'shape'):
            params = np.prod(layer.bias.shape)
            t, n = _cac_grad_params(params, layer.bias)
            tb_params += t
            ntb__params += n
            flops += oc
        return tb_params, ntb__params, flops

    else:
        raise NotImplementedError


@torch.no_grad()
def summary(model, x, return_results=False, extra_conv=(), extra_norm=(), extra_linear=()):
    """

    Args:
        model (nn.Module): model to summary
        x (torch.Tensor): input data
        return_results (bool): return results

    Returns:

    """
    # change bn work way
    model.eval()

    def register_hook(layer):
        def hook(layer, input, output):
            model_name = str(layer.__class__.__name__)
            module_idx = len(model_summary)
            s_key = '{}-{}'.format(model_name, module_idx + 1)
            model_summary[s_key] = OrderedDict()
            model_summary[s_key]['input_shape'] = list(input[0].shape)
            if isinstance(output, (tuple, list)):
                model_summary[s_key]['output_shape'] = [list(o.shape) for o in output]
            else:
                model_summary[s_key]['output_shape'] = list(output.shape)
            tb_params = 0
            ntb__params = 0
            flops = 0

            if isinstance(layer, (nn.Conv2d, *extra_conv)):
                tb_params, ntb__params, flops = _cac_conv(layer, input, output)
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm, *extra_norm)):
                tb_params, ntb__params, flops = _cac_xx_norm(layer, input, output)
            elif isinstance(layer, (nn.Linear, *extra_linear)):
                tb_params, ntb__params, flops = _cac_linear(layer, input, output)
            elif isinstance(layer, nn.MultiheadAttention):
                tb_params, ntb__params, flops = _cac_msa(layer, input, output)

            if hasattr(layer, 'num_param') and callable(getattr(layer, 'num_param')):
                assert tb_params == 0 and ntb__params == 0, 'params has been calculated by default func.'
                tb_params, ntb__params = layer.num_param(input, output)

            if hasattr(layer, 'flops') and callable(getattr(layer, 'flops')):
                assert flops == 0, 'flops has been calculated by default func.'
                flops = layer.flops(input, output)

            model_summary[s_key]['trainable_params'] = tb_params
            model_summary[s_key]['non_trainable_params'] = ntb__params
            model_summary[s_key]['params'] = tb_params + ntb__params
            model_summary[s_key]['flops'] = flops

        if not isinstance(layer, (nn.Sequential, nn.ModuleList, nn.Identity, nn.ModuleDict)):
            hooks.append(layer.register_forward_hook(hook))

    model_summary = OrderedDict()
    hooks = []
    model.apply(register_hook)
    try:
        model(x)
    except Exception as e:
        raise e
    finally:
        for h in hooks:
            h.remove()

    summary_str = ''
    summary_str += '-' * 80 + '\n'
    line_new = "{:>20}  {:>25} {:>15} {:>15}\n".format("Layer (type)", "Output Shape", "Params", "FLOPs(M+A) #")
    summary_str += line_new
    summary_str += '=' * 80 + '\n'
    total_params = 0
    trainable_params = 0
    total_flops = 0
    for layer in model_summary:
        line_new = "{:>20}  {:>25} {:>15} {:>15}\n".format(
            layer,
            str(model_summary[layer]['output_shape']),
            model_summary[layer]['params'],
            model_summary[layer]['flops'],
        )
        summary_str += line_new
        total_params += model_summary[layer]['params']
        trainable_params += model_summary[layer]['trainable_params']
        total_flops += model_summary[layer]['flops']

    param_str = _flops_str(total_params)
    flop_str = _flops_str(total_flops)
    flop_str_m = _flops_str(total_flops // 2)
    param_size = total_params * 4 / (1024**2)
    if return_results:
        return total_params, total_flops

    summary_str += '=' * 80 + '\n'
    summary_str += '        Total parameters: {:,}  {}\n'.format(total_params, param_str)
    summary_str += '    Trainable parameters: {:,}\n'.format(trainable_params)
    summary_str += 'Non-trainable parameters: {:,}\n'.format(total_params - trainable_params)
    summary_str += 'Total flops(M)  : {:,}  {}\n'.format(total_flops // 2, flop_str_m)
    summary_str += 'Total flops(M+A): {:,}  {}\n'.format(total_flops, flop_str)
    summary_str += '-' * 80 + '\n'
    summary_str += 'Parameters size (MB): {:.2f}'.format(param_size)
    return summary_str
