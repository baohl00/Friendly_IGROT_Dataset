import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 
import math
from typing import Dict, Tuple, Optional

class ImageSelfAttention(nn.Module):
    """ Self-attention module for CNN's feature map.
    Inspired by: Zhang et al., 2018 The self-attention mechanism in SAGAN.
    """
    def __init__(self, planes):
        super(ImageSelfAttention, self).__init__()
        inner = planes // 8
        self.conv_f = nn.Conv1d(planes, inner, kernel_size=1, bias=False)
        self.conv_g = nn.Conv1d(planes, inner, kernel_size=1, bias=False)
        self.conv_h = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        f = self.conv_f(x)
        g = self.conv_g(x)
        h = self.conv_h(x)
        sim_beta = torch.matmul(f.transpose(1, 2), g)
        beta = nn.functional.softmax(sim_beta, dim=1)
        o = torch.matmul(h, beta)
        return o

class LayerNorm(nn.Module):
    """Layer norm used in Transformer layers."""

    def __init__(self, dim: int = 1, epsilon: float = 1e-6, use_scale: bool = True, use_bias: bool = True):
        super(LayerNorm, self).__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.use_bias = use_bias
        if self.use_scale:
            self.scale = nn.Parameter(torch.ones(dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normed_x = (x - mean) / torch.sqrt(var + self.epsilon)

        if self.use_scale:
            normed_x = normed_x * (1 + self.scale)
        if self.use_bias:
            normed_x = normed_x + self.bias

        return normed_x

class Weight(nn.Module):
    def __init__(self, input_dim: int = 0, hidden_dim: int = 0, param_dtype: torch.dtype = torch.float32):
        super(Weight, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.param_dtype = param_dtype
        self.w = nn.Parameter(torch.empty(input_dim, hidden_dim, dtype=param_dtype))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w)

class Bias(nn.Module):
    def __init__(self, hidden_dim: int = 0, param_dtype: torch.dtype = torch.float32):
        super(Bias, self).__init__()
        self.hidden_dim = hidden_dim
        self.param_dtype = param_dtype
        self.b = nn.Parameter(torch.zeros(hidden_dim, dtype=param_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b
    
class FFN(nn.Module):
    """Feed-forward network."""
    def __init__(self, input_dim: int = 0, output_dim: int = 0, use_bias: bool = True, dropout: float = 0.5, use_activation: bool = True):
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout)
        self.linear = Weight(input_dim, output_dim)
        if use_bias:
            self.bias = Bias(output_dim)
        if use_activation:
            self.activation = nn.GELU() # nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.use_bias:
            x = self.bias(x)
        x = self.dropout(x)
        if self.use_activation:
            x = self.activation(x)
        return x

class TransformerFFN(nn.Module):
    """Feed-forward network used in Transformer layers with residual connection."""
    def __init__(self, input_dim: int = 0, output_dim: int = 0, hidden_dim: int = 0, use_bias: bool = True, add_skip_connection: bool = True):
        super(TransformerFFN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim != 0 else input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.add_skip_connection = add_skip_connection
        self.ln = LayerNorm(dim=input_dim)
        self.ffn1 = FFN(input_dim=input_dim, output_dim=hidden_dim, use_bias=use_bias)
        self.ffn2 = FFN(input_dim=hidden_dim, output_dim=self.output_dim, use_bias=use_bias, use_activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.ffn1(x)
        x = self.ffn2(x)
        if self.add_skip_connection:
            x = x + residual
        return x
    
class AttentionProjection(nn.Module):
    """Projection (e.g., k) used in self-attention.

    output_proj: Whether it is out projection or not. If False, we use
      "...D,DNH->...NH" for query,key,value projection. Otherwise we use
      "...NH,DNH->...D" for output projection.
    """
    def __init__(self, input_dim: int = 0, num_heads: int = 0, dim_per_head: int = 0, use_bias: bool = True, output_proj: bool = False, param_dtype: torch.dtype = torch.float32):
        super(AttentionProjection, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.use_bias = use_bias
        self.output_proj = output_proj
        self.param_dtype = param_dtype
        pc_shape = [input_dim, num_heads, dim_per_head] if not output_proj else [num_heads, dim_per_head, input_dim]
        self.w = nn.Parameter(torch.empty(pc_shape, dtype=param_dtype))
        self.reset_parameters()
        if use_bias:
            if output_proj:
                self.b = nn.Parameter(torch.zeros(input_dim, dtype=param_dtype))
            else:
                self.b = nn.Parameter(torch.zeros(num_heads, dim_per_head, dtype=param_dtype))

    def reset_parameters(self):
        init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_proj:
            ret = torch.einsum('...NH,NHD->...D', x, self.w)
        else:
            ret = torch.einsum('...D,DNH->...NH', x, self.w)
        if self.use_bias:
            ret += self.b
        return ret

class PerDimScale(nn.Module):
        def __init__(self, dim):
            super(PerDimScale, self).__init__()
            self.dim = dim
            self.per_dim_scale = nn.Parameter(torch.ones(dim))
            
        def forward(self, x):
            assert x.shape[-1] == self.dim, f'Input last dimension {x.shape[-1]} does not match expected dimension {self.dim}'
            r_softplus_0 = 1.442695041
            scale = r_softplus_0 / (self.dim ** 0.5)
            scale = scale * F.softplus(self.per_dim_scale)
            return x * scale

class DotProductAttention(nn.Module):
    """Self-attention used in Transformer layers."""
    def __init__(self, input_dim: int = 0, hidden_dim: int = 0, num_heads: int = 1, use_bias: bool = True, dim_per_head: int = 0, use_per_dim_scale: bool = False):
        super(DotProductAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.dim_per_head = dim_per_head if dim_per_head != 0 else hidden_dim // num_heads
        self.use_per_dim_scale = use_per_dim_scale
        self.key = AttentionProjection(input_dim=input_dim, num_heads=num_heads, dim_per_head=self.dim_per_head, use_bias=use_bias)
        self.query = AttentionProjection(input_dim=input_dim, num_heads=num_heads, dim_per_head=self.dim_per_head, use_bias=use_bias)
        self.value = AttentionProjection(input_dim=input_dim, num_heads=num_heads, dim_per_head=self.dim_per_head, use_bias=use_bias)
        if use_per_dim_scale:
            self.per_dim_scale = PerDimScale(dim=self.dim_per_head)
        self.post = AttentionProjection(input_dim=input_dim, num_heads=num_heads, dim_per_head=self.dim_per_head, use_bias=use_bias, output_proj=True)

    def _dot_atten(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dot-product attention."""
        if self.use_per_dim_scale:
            query = self.per_dim_scale(query)
        else:
            query = query * (self.dim_per_head ** -0.5)
        logits = torch.einsum('BTNH,BSNH->BNTS', query, key)
        cap = torch.tensor(50.0, dtype=logits.dtype)
        logits = cap * torch.tanh(logits / cap)
        probs = torch.softmax(logits, dim=-1).type_as(key)
        encoded = torch.einsum('BNTS,BSNH->BTNH', probs, value)
        return encoded, probs

    def forward(self, q_vector: torch.Tensor, k_vector: torch.Tensor, v_vector: torch.Tensor, atten_mask: None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        query_proj = self.query(q_vector)
        key_proj = self.key(k_vector)
        value_proj = self.value(v_vector)
        encoded, atten_probs = self._dot_atten(query_proj, key_proj, value_proj)
        encoded = self.post(encoded)
        return encoded, atten_probs

class Transformer(nn.Module):
    """Transformer layer used in multimodal encoder."""
    def __init__(self, num_heads: int, input_dim: int = 0, hidden_dim: int = 0, output_dim: int = 0, use_bias: bool = True, add_skip_connection: bool = True, use_per_dim_scale: bool = False):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim != 0 else input_dim
        self.use_bias = use_bias
        self.add_skip_connection = add_skip_connection
        self.use_per_dim_scale = use_per_dim_scale
        self.ff_layer = TransformerFFN(input_dim=input_dim, output_dim=self.output_dim, hidden_dim=hidden_dim, use_bias=use_bias, add_skip_connection=add_skip_connection)
        self.self_attention = DotProductAttention(input_dim=input_dim, hidden_dim=input_dim, num_heads=num_heads, use_bias=use_bias, use_per_dim_scale=use_per_dim_scale)
        self.layer_norm = LayerNorm(dim=input_dim)

    def forward(self, x: torch.Tensor, attn_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_normalized = self.layer_norm(x)
        atten_output, atten_probs = self.self_attention(x_normalized, x_normalized, x_normalized)
        if self.add_skip_connection:
            atten_output += x
        output = self.ff_layer(atten_output)
        return output, atten_probs

class StackedTransformer(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, input_dim: int, hidden_dim: int, use_bias: bool = True, add_skip_connection: bool = True, use_per_dim_scale: bool = False):
        super(StackedTransformer, self).__init__()
        assert num_layers > 0
        assert input_dim > 0
        assert hidden_dim > 0
        assert num_heads > 0
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            Transformer(num_heads=num_heads, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, use_bias=use_bias, add_skip_connection=add_skip_connection, use_per_dim_scale=use_per_dim_scale) 
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x, attn_mask)
        return x
    
class AttenTokenPoolingLayer(nn.Module):
    def __init__(self, input_dim: int = 0, query_dim: Optional[int] = None, hidden_dim: int = 0, num_heads: int = 1, num_query_tokens: int = 1, use_bias: bool = True, use_per_dim_scale: bool = True, param_dtype: torch.dtype = torch.float32, kernel_init: callable = nn.init.kaiming_normal_):
        super(AttenTokenPoolingLayer, self).__init__()
        assert input_dim > 0, 'input_dim must be positive'
        self.input_dim = input_dim
        self.query_dim = query_dim or input_dim
        self.hidden_dim = hidden_dim if hidden_dim > 0 else 4 * input_dim
        self.num_heads = num_heads
        self.num_query_tokens = num_query_tokens
        self.use_bias = use_bias
        self.use_per_dim_scale = use_per_dim_scale
        self.param_dtype = param_dtype
        self.pool_attn = DotProductAttention(input_dim=input_dim, hidden_dim=self.hidden_dim, num_heads=num_heads, use_bias=use_bias, use_per_dim_scale=use_per_dim_scale)
        self.pool_attn_ln = LayerNorm(dim=self.query_dim, epsilon=1e-6)
        self.pooling_attn_query = nn.Parameter(torch.empty((num_query_tokens, self.query_dim), dtype=param_dtype))
        kernel_init(self.pooling_attn_query)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        batch_size, _ = embeds.shape[:2]
        query = self.pooling_attn_query.unsqueeze(0).expand(batch_size, -1, -1)
        key = embeds
        pooled_output, _ = self.pool_attn(query, key, embeds)
        pooled_output = self.pool_attn_ln(pooled_output)
        #print(query.shape, key.shape, pooled_output.shape)
        return pooled_output
