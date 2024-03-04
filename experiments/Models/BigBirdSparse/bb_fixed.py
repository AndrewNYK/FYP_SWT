# -*- coding: utf-8 -*-
from typing import Optional, Tuple, List
import warnings

import torch
from torch import Tensor
from torch.nn.functional import _in_projection, _in_projection_packed, linear, dropout
from torch.nn.functional import pad as pad
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.parameter import Parameter

from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

import math

def _fixed_sparse_attention(
    # q: Tensor,
    # k: Tensor,
    v: Tensor,
    # bq: Tensor,
    # bk: Tensor,
    bv: Tensor,
    bsz: int,
    num_heads: int,
    from_seq_len: int,
    to_block_size: int,
    head_dim: int,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0
) -> Tuple[Tensor, Tensor]:

    rsqrt_d = 1/math.sqrt(head_dim)
    num_blocks = from_seq_len//to_block_size
    # prod_1 = ndbmm_t(bq[:,:,0],k, 4)
    # prod_1 = prod_1 * rsqrt_d
    # attn_w1 = torch.nn.functional.softmax(prod_1, dim=-1)
    attn_w1 = (1/from_seq_len)*torch.ones((bsz,num_heads,to_block_size,from_seq_len),device=v.device) #(bsz, nheads, block_size, from_seq_len)
    if dropout_p > 0.0:
        attn_w1 = dropout(attn_w1,dropout_p)
    ctx_1 = ndbmm(attn_w1,v,nd=4)
    ctx_1.unsqueeze_(2)
    
    
    # partial_k_2 = torch.cat([bk[:,:,0],
    #                          bk[:,:,1],
    #                          bk[:,:,2],
    #                          bk[:,:,-1]], dim=2)
    partial_v_2 = torch.cat([bv[:,:,0],
                              bv[:,:,1],
                              bv[:,:,2],
                              bv[:,:,-1]], dim=2)
    # prod_2 = ndbmm_t(bq[:,:,1],partial_k_2, 4)
    # prod_2 = prod_2 * rsqrt_d
    # attn_w2 = torch.nn.functional.softmax(prod_2, dim=-1)
    attn_w2 = (1/(4*to_block_size))*torch.ones((bsz,num_heads,to_block_size,4*to_block_size),
                                               device=partial_v_2.device)
    if dropout_p > 0.0:
        attn_w2 = dropout(attn_w2,dropout_p)
    ctx_2 = ndbmm(attn_w2,partial_v_2,nd=4)
    ctx_2.unsqueeze_(2)
    
    
    # expanded_bk = torch.cat([bk[:,:,1:-3],
    #                          bk[:,:,2:-2],
    #                          bk[:,:,3:-1]], dim=3)
    expanded_bv = torch.cat([bv[:,:,1:-3],
                             bv[:,:,2:-2],
                             bv[:,:,3:-1]], dim=3)
    # m_band_prod = ndbmm_t(bq[:,:,2:-2],expanded_bk,nd=5)
    # l_band_prod = torch.einsum("bhlqd,bhkd->bhlqk", bq[:,:,2:-2], bk[:,:,0])
    # r_band_prod = torch.einsum("bhlqd,bhkd->bhlqk", bq[:,:,2:-2], bk[:,:,-1])
    
    # m_band_prod = m_band_prod*rsqrt_d
    # l_band_prod = l_band_prod*rsqrt_d
    # r_band_prod = r_band_prod*rsqrt_d
    
    # band_prod = torch.cat([l_band_prod,m_band_prod,r_band_prod],dim=-1)
    # attn_w3 = torch.nn.functional.softmax(band_prod, dim=-1)
    attn_w3 = (1/(5*to_block_size))*torch.ones((bsz,num_heads,num_blocks-4,to_block_size,5*to_block_size),
                                               device=expanded_bv.device)
    if dropout_p > 0.0:
        attn_w3 = dropout(attn_w3,dropout_p)
    ctx_3 = ndbmm(attn_w3[:, :, :, :, to_block_size : 4 * to_block_size], expanded_bv, nd=5)
    ctx_3 += torch.einsum("bhlqk,bhkd->bhlqd", attn_w3[:, :, :, :, :to_block_size], bv[:, :, 0])
    ctx_3 += torch.einsum("bhlqk,bhkd->bhlqd", attn_w3[:, :, :, :, -to_block_size:], bv[:, :, -1])
    
    # partial_k_4 = torch.cat([bk[:,:,0],
    #                          bk[:,:,-3],
    #                          bk[:,:,-2],
    #                          bk[:,:,-1]], dim=2)
    partial_v_4 = torch.cat([bv[:,:,0],
                             bv[:,:,-3],
                             bv[:,:,-2],
                             bv[:,:,-1]], dim=2)
    # prod_4 = ndbmm_t(bq[:,:,-2],partial_k_4, 4)
    # prod_4 = prod_4 * rsqrt_d
    # attn_w4 = torch.nn.functional.softmax(prod_4, dim=-1)
    attn_w4 = (1/(4*to_block_size))*torch.ones((bsz,num_heads,to_block_size,4*to_block_size),device=partial_v_4.device)
    if dropout_p > 0.0:
        attn_w4 = dropout(attn_w4,dropout_p)
    ctx_4 = ndbmm(attn_w4,partial_v_4,nd=4)
    ctx_4.unsqueeze_(2)
    
    # prod_5 = ndbmm_t(bq[:,:,-1],k,nd=4)
    # prod_5 = prod_5 * rsqrt_d
    # attn_w5 = torch.nn.functional.softmax(prod_5, dim=-1)
    attn_w5 = (1/from_seq_len)*torch.ones((bsz,num_heads,to_block_size,from_seq_len),device=v.device)
    if dropout_p > 0.0:
        attn_w5 = dropout(attn_w5,dropout_p)
    ctx_5 = ndbmm(attn_w5,v,nd=4)
    ctx_5.unsqueeze_(2)
    
    ctx = torch.cat([ctx_1,ctx_2,ctx_3,ctx_4,ctx_5],dim=2)
    ctx = ctx.view((bsz,num_heads,from_seq_len,-1))
    ctx = torch.transpose(ctx,1,2)
    
    ctx = ctx.contiguous().view((bsz,from_seq_len,-1))
    return ctx, torch.empty((0))


def ndbmm(a: Tensor, b: Tensor, nd: int):
    return torch.bmm(a.reshape((-1,) + a.shape[-2:]), b.reshape((-1,) + b.shape[-2:])).view(
            a.shape[: nd - 2] + (a.shape[nd - 2], b.shape[nd - 1]))

def ndbmm_t(a: Tensor, b: Tensor,nd: int):
    return torch.bmm(
            a.reshape((-1,) + a.shape[-2:]), b.reshape((-1,) + b.shape[-2:]).transpose(1, 2)
        ).view(a.shape[: nd - 2] + (a.shape[nd - 2], b.shape[nd - 2]))

def score_transpose(x: Tensor, num_heads: int, head_dim: int) -> Tensor:
    #print(x.size())
    #new_shape = x.size()[:-1] + (num_heads, head_dim)
    new_shape = (x.size()[0]//num_heads,num_heads,x.size()[1],x.size()[2])
    x = x.view(*new_shape)
    return x #x.permute(0,2,1,3) #x.permute(1,2,0,3)


def f_in_projection_packed(
    # q: Tensor,
    # k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is the projection tensor value. Modified variant that computes only the value tensor.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = v.size(-1)
    # if k is v:
    #     if q is k:
    #         # self-attention
    #         return linear(q, w, b).chunk(3, dim=-1)
    #     else:
    #         # encoder-decoder attention
    #         w_q, w_kv = w.split([E, E * 2])
    #         if b is None:
    #             b_q = b_kv = None
    #         else:
    #             b_q, b_kv = b.split([E, E * 2])
    #         return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    # else:
    #     w_q, w_k, w_v = w.chunk(3)
    #     if b is None:
    #         b_q = b_k = b_v = None
    #     else:
    #         b_q, b_k, b_v = b.chunk(3)
    #     return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
    return linear(v,w,b)


def f_in_projection(
    v: Tensor,
    w_v: Tensor,
    # b_q: Optional[Tensor] = None,
    # b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Ev = v.size(-1)
    assert b_v is None or b_v.shape == (Ev,), f"expecting value bias shape of {(Ev,)}, but got {b_v.shape}"
    return linear(v, w_v, b_v)


def sparse_fixed_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    block_size: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    # bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    # q_proj_weight: Optional[Tensor] = None,
    # k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    # static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        # q, k, v = f_in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        v = f_in_projection_packed(value, in_proj_weight, in_proj_bias)
    else:
        # assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        # assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_v = None
        else:
            b_v = in_proj_bias
        v = f_in_projection(value, v_proj_weight, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_v is not None:
        # assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        # k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        # assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    #print('qsha', q.shape)
    # q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # #print(q.shape)
    # if static_k is None:
    #     k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    # else:
    #     # TODO finish disentangling control flow so we don't do in-projections when statics are passed
    #     assert static_k.size(0) == bsz * num_heads, \
    #         f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
    #     assert static_k.size(2) == head_dim, \
    #         f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
    #     k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        # k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = v.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0
    

    from_seq_len = src_len
    to_seq_len = tgt_len
    from_block_size = to_block_size = block_size
    assert from_seq_len % from_block_size == 0, "Sequence length must be divisible by block size."
    assert to_seq_len % to_block_size == 0, "Sequence length must be divisible by block size."
    assert from_seq_len//block_size > 4, "Block size is too large given sequence length, use full attention."
    #Reshape again for sparse calculation
    # q = score_transpose(q,num_heads,head_dim)
    # k = score_transpose(k,num_heads,head_dim)
    v = score_transpose(v,num_heads,head_dim)
    # blocked_query_matrix = q.view(bsz, num_heads, from_seq_len // from_block_size, from_block_size, -1)
    # blocked_key_matrix = k.view(bsz, num_heads, to_seq_len // to_block_size, to_block_size, -1)
    blocked_value_matrix = v.view(bsz, num_heads, to_seq_len // to_block_size, to_block_size, -1)

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _fixed_sparse_attention(v,
                                                               blocked_value_matrix,
                                                               bsz, num_heads, from_seq_len,
                                                               to_block_size, head_dim,
                                                               attn_mask, dropout_p
                                                               )
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

#Fixed pattern 'attention' that skips computing query x key products
class BlockSparseFixedAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, block_size, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BlockSparseFixedAttention,self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv,
                                                        add_zero_attn, kdim, vdim, batch_first, device, dtype)
        self.block_size = block_size
        
        #Delete extraneous parameters related to query and key
        del(self.q_proj_weight)
        del(self.k_proj_weight)
        #Resize in_projection_weight
        if self.in_proj_weight is not None:
            self.in_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        if self.in_proj_bias is not None:
            self.in_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
        
        del(self.bias_k)
        self._reset_parameters()
    
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
            
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = sparse_fixed_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, self.block_size,
                self.in_proj_weight, self.in_proj_bias,
                # self.bias_k,
                self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = sparse_fixed_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, self.block_size,
                self.in_proj_weight, self.in_proj_bias,
                # self.bias_k,
                self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights