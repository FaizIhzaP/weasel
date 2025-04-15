from torch import nn
from collections import OrderedDict
from math import ceil
import torch.nn.functional as F
import re
import warnings

from torch.nn import DataParallel as DataParallel_
from torch.nn.parallel import parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch.nn.parallel.replicate import _broadcast_coalesced_reshape
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.batchnorm import _BatchNorm

# ---- 1. Module for meta ----
class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """
    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None

        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names

            else:
                key_escape = re.escape(key)
                key_re = re.compile(r'^{0}\.(.+)'.format(key_escape))

                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r'\1', k) for k in all_names if key_re.match(k) is not None]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn('Module `{0}` has no parameter corresponding to the '
                          'submodule named `{1}` in the dictionary `params` '
                          'provided as an argument to `forward()`. Using the '
                          'default parameters for this submodule. The list of '
                          'the parameters in `params`: [{2}].'.format(
                          self.__class__.__name__, key, ', '.join(all_names)),
                          stacklevel=2)
            return None

        return OrderedDict([(name, params[f'{key}.{name}']) for name in names])

# ---- 2. Sparse for meta ----
class MetaEmbedding(nn.Embedding, MetaModule):
    __doc__ = nn.Embedding.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        return F.embedding(
            input, params['weight'], self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

class MetaEmbeddingBag(nn.EmbeddingBag, MetaModule):
    __doc__ = nn.EmbeddingBag.__doc__

    def forward(self, input, offsets=None, per_sample_weights=None, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        return F.embedding_bag(input, params['weight'], offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode, self.sparse,
                               per_sample_weights, self.include_last_offset)

# ---- 3. Parallel for meta ----
class DataParallel(DataParallel_, MetaModule):
    __doc__ = DataParallel_.__doc__

    def scatter(self, inputs, kwargs, device_ids):
        if not isinstance(self.module, MetaModule):
            return super(DataParallel, self).scatter(inputs, kwargs, device_ids)

        params = kwargs.pop('params', None)
        inputs_, kwargs_ = scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
        # Add params argument unchanged back in kwargs
        replicas = self._replicate_params(params, inputs_, device_ids,
                                          detach=not torch.is_grad_enabled())
        kwargs_ = tuple(dict(params=replica, **kwarg)
                        for (kwarg, replica) in zip(kwargs_, replicas))
        return inputs_, kwargs_

    def _replicate_params(self, params, inputs, device_ids, detach=False):
        if params is None:
            module_params = OrderedDict(self.module.named_parameters())
        else:
            # Temporarily disable the warning if no parameter with key prefix
            # `module` was found. In that case, the original params dictionary
            # is used.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                module_params = self.get_subdict(params, key='module')
            if module_params is None:
                module_params = params

        replicas = _broadcast_coalesced_reshape(list(module_params.values()),
                                                device_ids[:len(inputs)],
                                                detach)
        replicas = tuple(OrderedDict(zip(module_params.keys(), replica))
                         for replica in replicas)
        return replicas

# ---- 4. Normalization for meta ----
class MetaLayerNorm(nn.LayerNorm, MetaModule):
    __doc__ = nn.LayerNorm.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        weight = params.get('weight', None)
        bias = params.get('bias', None)
        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)

# ---- 5. Linear for meta ----
class MetaLinear(nn.Linear, MetaModule):
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.linear(input, params['weight'], bias)

class MetaBilinear(nn.Bilinear, MetaModule):
    __doc__ = nn.Bilinear.__doc__

    def forward(self, input1, input2, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.bilinear(input1, input2, params['weight'], bias)

# ---- 6. Convolution for meta ----
class MetaConv1d(nn.Conv1d, MetaModule):
    __doc__ = nn.Conv1d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return self._conv_forward(input, params['weight'], bias)


class MetaConv2d(nn.Conv2d, MetaModule):
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return self._conv_forward(input, params['weight'], bias)


class MetaConv3d(nn.Conv3d, MetaModule):
    __doc__ = nn.Conv3d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return self._conv_forward(input, params['weight'], bias)

# ---- 7. Container for meta ----
class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=self.get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError('The module must be either a torch module '
                    '(inheriting from `nn.Module`), or a `MetaModule`. '
                    'Got type: `{0}`'.format(type(module)))
        return input

# ---- 8. BatchNorm for meta ----
class _MetaBatchNorm(_BatchNorm, MetaModule):
    def forward(self, input, params=None):
        self._check_input_dim(input)
        if params is None:
            params = OrderedDict(self.named_parameters())

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        weight = params.get('weight', None)
        bias = params.get('bias', None)

        return F.batch_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

class MetaBatchNorm1d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm1d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

class MetaBatchNorm2d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm2d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class MetaBatchNorm3d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm3d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

# ---- 9. Activation for meta ----
class MetaMultiheadAttention(nn.MultiheadAttention, MetaModule):
    __doc__ = nn.MultiheadAttention.__doc__

    def __init__(self, *args, **kwargs):
        super(MetaMultiheadAttention, self).__init__(*args, **kwargs)
        factory_kwargs = {
            'device': kwargs.get('device', None),
            'dtype': kwargs.get('dtype', None)
        }
        bias = kwargs.get('bias', True)
        self.out_proj = MetaLinear(self.embed_dim, self.embed_dim,
            bias=bias, **factory_kwargs)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        in_proj_weight = params.get('in_proj_weight', None)
        in_proj_bias = params.get('in_proj_bias', None)
        out_proj_bias = params.get('out_proj.bias', None)
        bias_k = params.get('bias_k', None)
        bias_v = params.get('bias_v', None)

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, params['out_proj.weight'], out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=params['q_proj_weight'],
                k_proj_weight=params['k_proj_weight'],
                v_proj_weight=params['v_proj_weight'])
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, params['out_proj.weight'], out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
        

