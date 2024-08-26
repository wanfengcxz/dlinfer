import math
import torch
import torch_mlu
from bangtransformer.torch import bt_ops

from infer_ext.vendor import vendor_ops_registry
from infer_ext.utils.registry import register_ops
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence, Tuple

__all__ =[
    "add_rms_norm",
    "apply_rotary_pos_emb",
    "context_attention",
    "fill_kv_cache",
    "paged_decode_attention",
    "paged_prefill_attention",
    "rms_norm",
    "moe_gating_topk_softmax",
    "get_cache_len",
]

@register_ops(vendor_ops_registry)
def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float
) -> Tensor:
    assert 1 < hidden_states.ndim < 4, "only support hidden_states: [total_seq_len, head_size], [batch_size, seq_lens, head_size]"
    
    hidden_states = hidden_states.contiguous()
    shape = hidden_states.shape
    hidden_states = hidden_states.view(-1, shape[-1])
    store_output_before_norm = False
    normed_hidden_states = bt_ops.fused_rms_norm(hidden_states, None, weight, None, None, epsilon, store_output_before_norm)[0]
    normed_hidden_states = normed_hidden_states.view(shape)
    return normed_hidden_states

@register_ops(vendor_ops_registry)
def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    assert 1 < hidden_states.ndim < 4, "only support hidden_states: [total_seq_len, head_size], [batch_size, seq_lens, head_size]"
    
    shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, shape[-1])
    residual = residual.reshape(-1, shape[-1])
    
    store_output_before_norm = True
    normed_hidden_states, added_hidden_states = \
        bt_ops.fused_rms_norm(hidden_states, residual, weight, None, None, epsilon, store_output_before_norm)
    
    normed_hidden_states = normed_hidden_states.reshape(shape)
    added_hidden_states = added_hidden_states.reshape(shape)
    return normed_hidden_states, added_hidden_states

@register_ops(vendor_ops_registry)
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    position_ids: Optional[Tensor],
    cos_full: Optional[Tensor],
    sin_full: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    assert query.ndim == 3, "only support q:[totalSeq, head ,head_dim]"
    assert key.ndim == 3, "only support k:[totalSeq, head ,head_dim]"
    interleaved = False
    embeded_query = torch.empty_like(query)
    embeded_key = torch.empty_like(key)
    if position_ids is not None:
        cos = cos_full[position_ids]
        sin = sin_full[position_ids]
    #view totalSeq as a long sequence
    cu_seq_lens = torch.Tensor([0,query.shape[0]]).long().mlu()
    max_context_len = query.shape[0]
    bt_ops.apply_rotary(embeded_query, query, sin, cos, position_ids, cu_seq_lens, interleaved, False, False, max_context_len)
    bt_ops.apply_rotary(embeded_key, key, sin, cos, position_ids, cu_seq_lens, interleaved, False, False, max_context_len)
    return embeded_query,embeded_key

@register_ops(vendor_ops_registry)
def fill_kv_cache(
    key: Tensor,
    value: Tensor,     
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    assert key.ndim == 3 and value.ndim == 3, \
        "only support key, value: [total_seq_len, head_num, head_size]"
    assert key_cache.ndim == 4 and value_cache.ndim == 4, \
        "only support key_cache, value_cache: [block_num, head_num, block_size, head_size]"
    assert kv_indices.ndim == 1, "only support kv_indices: [total_seq_len]"
    
    # only support contiguous k,v
    key = key.contiguous()
    value = value.contiguous()

    bt_ops.reshape_paged_cache(key, value, key_cache, value_cache, kv_indices)
    return key_cache, value_cache

@register_ops(vendor_ops_registry)
def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Optional[Tensor],
    block_size: int,
    kv_seq_len: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_qk_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
) -> Tensor:
    assert query.ndim == 4, "only support q:[batch,seq_q=1, head ,head_dim]"
    assert query.shape[1] == 1, "only support seq_q = 1 in paged decode attention"
    assert key_cache.ndim == 4, "only support k_cache:[num_blocks, kv_head_num, block_size, head_size]"
    assert value_cache.ndim == 4, "only support v_cache:[num_blocks, kv_head_num, block_size, head_size]"
    assert block_table.ndim == 2, "only support bloack_table:[batch_size, max_num_blocks_per_seq]"
    batch_size = block_table.shape[0]
    dim = query.shape[3]
    k_cache_quant_scale = None
    v_cache_quant_scale = None
    max_context_lens = torch.max(kv_seq_len)
    softmax_scale = 1. / math.sqrt(dim)

    out = attn_output.view_as(query)

    bt_ops.single_query_cached_kv_attn(query, key_cache, value_cache, block_table, kv_seq_len,k_cache_quant_scale, v_cache_quant_scale, alibi_slopes, out, max_context_lens, 0, 0, softmax_scale)

if __name__ == '__main__':
    pass
