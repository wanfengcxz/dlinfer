# Copyright (c) 2026, DeepLink. All rights reserved.

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

pytestmark = pytest.mark.lmdeploy

if not torch.npu.is_available():
    pytest.skip("Ascend NPU is required", allow_module_level=True)

from dlinfer.vendor.ascend.attention import decode_attention_mla
from dlinfer.vendor.ascend.torch_npu_ops import prefill_attention

DTYPE = torch.bfloat16
DEVICE = torch.device("npu")
FAI_CAUSAL_MASK_SIZE = 2048
NUM_Q_HEADS = 16
NUM_KV_HEADS = 1
STANDARD_HEAD_DIM = 128
STANDARD_SOFTMAX_SCALE = 1.0 / math.sqrt(STANDARD_HEAD_DIM)
DENSE_NOPE_HEAD_DIM = 128
DENSE_ROPE_HEAD_DIM = 64
DENSE_QK_HEAD_DIM = DENSE_NOPE_HEAD_DIM + DENSE_ROPE_HEAD_DIM
DENSE_V_HEAD_DIM = DENSE_NOPE_HEAD_DIM
DENSE_SOFTMAX_SCALE = 1.0 / math.sqrt(DENSE_QK_HEAD_DIM)
MLA_NOPE_HEAD_DIM = 512
MLA_ROPE_HEAD_DIM = 64
MLA_QK_HEAD_DIM = MLA_NOPE_HEAD_DIM + MLA_ROPE_HEAD_DIM
MLA_V_HEAD_DIM = MLA_NOPE_HEAD_DIM
# DeepSeekV2 keeps the scale of its pre-absorption 128 + 64 query head.
MLA_SOFTMAX_SCALE = 1.0 / math.sqrt(128 + MLA_ROPE_HEAD_DIM)


@pytest.fixture(scope="module")
def fai_causal_mask():
    """Build the fixed split-fuse mask expected by FAI sparse mode 3."""
    return torch.triu(
        torch.ones(
            FAI_CAUSAL_MASK_SIZE,
            FAI_CAUSAL_MASK_SIZE,
            dtype=torch.int8,
            device=DEVICE,
        ),
        diagonal=1,
    )


def _randn(shape):
    return torch.randn(shape, dtype=torch.float32).to(DTYPE)


def _repeat_kv(hidden_states, num_q_heads):
    num_kv_heads = hidden_states.shape[1]
    assert num_q_heads % num_kv_heads == 0
    return hidden_states.repeat_interleave(num_q_heads // num_kv_heads, dim=1)


def _torch_prefill_attention(query, key, value, seq_lens, softmax_scale):
    outputs = []
    start = 0
    for seq_len in seq_lens:
        end = start + seq_len
        q = query[start:end].float().transpose(0, 1)
        k = _repeat_kv(key[start:end], NUM_Q_HEADS).float().transpose(0, 1)
        v = _repeat_kv(value[start:end], NUM_Q_HEADS).float().transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-1, -2)) * softmax_scale
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(causal_mask, float("-inf"))
        outputs.append(torch.matmul(torch.softmax(scores, dim=-1), v).transpose(0, 1))
        start = end

    return torch.cat(outputs)


def _torch_decode_attention(query, key_cache, block_table, kv_seq_lens):
    outputs = []
    block_size = key_cache.shape[1]
    for batch_idx, kv_seq_len in enumerate(kv_seq_lens):
        num_blocks = math.ceil(kv_seq_len / block_size)
        block_ids = block_table[batch_idx, :num_blocks].long()
        key = key_cache[block_ids].flatten(0, 1)[:kv_seq_len]
        value = key[..., :MLA_V_HEAD_DIM]

        q = query[batch_idx].float()
        k = _repeat_kv(key, NUM_Q_HEADS).float().transpose(0, 1)
        v = _repeat_kv(value, NUM_Q_HEADS).float().transpose(0, 1)
        scores = torch.matmul(q.unsqueeze(1), k.transpose(-1, -2))
        scores = scores * MLA_SOFTMAX_SCALE
        outputs.append(torch.matmul(torch.softmax(scores, dim=-1), v).squeeze(1))

    return torch.stack(outputs)


def _assert_prefill_attention_matches_torch(
    qk_head_dim, value_head_dim, softmax_scale, causal_mask
):
    seq_lens = [112, 70, 31]
    num_tokens = sum(seq_lens)

    query = _randn((num_tokens, NUM_Q_HEADS, qk_head_dim))
    key = _randn((num_tokens, NUM_KV_HEADS, qk_head_dim))
    value = _randn((num_tokens, NUM_KV_HEADS, value_head_dim))
    expected = _torch_prefill_attention(query, key, value, seq_lens, softmax_scale)

    query = query.to(DEVICE)
    key = key.to(DEVICE)
    value = value.to(DEVICE)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    max_seq_len = max(seq_lens)
    output = torch.empty(
        (num_tokens, NUM_Q_HEADS, value_head_dim), dtype=DTYPE, device=DEVICE
    )

    actual = prefill_attention(
        query=query,
        key=key,
        value=value,
        q_start_loc=None,
        q_seq_len=seq_lens_tensor,
        max_q_seq_len=max_seq_len,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        attn_mask=[causal_mask],
        softmax_scale=softmax_scale,
        alibi_slopes=None,
        attn_output=output,
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=5e-3, atol=5e-3)


def test_prefill_attention_standard_matches_torch(fai_causal_mask):
    torch.manual_seed(20260812)
    _assert_prefill_attention_matches_torch(
        qk_head_dim=STANDARD_HEAD_DIM,
        value_head_dim=STANDARD_HEAD_DIM,
        softmax_scale=STANDARD_SOFTMAX_SCALE,
        causal_mask=fai_causal_mask,
    )


def test_prefill_attention_dense_matches_torch(fai_causal_mask):
    torch.manual_seed(20260813)
    _assert_prefill_attention_matches_torch(
        qk_head_dim=DENSE_QK_HEAD_DIM,
        value_head_dim=DENSE_V_HEAD_DIM,
        softmax_scale=DENSE_SOFTMAX_SCALE,
        causal_mask=fai_causal_mask,
    )


def test_prefill_attention_mla_matches_torch(fai_causal_mask):
    torch.manual_seed(20260814)
    _assert_prefill_attention_matches_torch(
        qk_head_dim=MLA_QK_HEAD_DIM,
        value_head_dim=MLA_V_HEAD_DIM,
        softmax_scale=MLA_SOFTMAX_SCALE,
        causal_mask=fai_causal_mask,
    )


def test_decode_attention_mla_matches_torch():
    torch.manual_seed(20260813)
    batch_size = 3
    block_size = 128
    num_blocks = 6
    kv_seq_lens = [130, 77, 17]
    block_table = torch.tensor(
        [[4, 1], [3, 0], [5, 2]],
        dtype=torch.int32,
    )

    query = _randn((batch_size, NUM_Q_HEADS, MLA_QK_HEAD_DIM))
    key_cache = _randn((num_blocks, block_size, NUM_KV_HEADS, MLA_QK_HEAD_DIM))
    expected = _torch_decode_attention(query, key_cache, block_table, kv_seq_lens)

    query = query.to(DEVICE)
    key_cache = key_cache.to(DEVICE)
    output = torch.empty(
        (batch_size, NUM_Q_HEADS, MLA_V_HEAD_DIM), dtype=DTYPE, device=DEVICE
    )

    actual = decode_attention_mla(
        query=query,
        key_cache=key_cache,
        num_kv_heads=NUM_KV_HEADS,
        num_q_heads=NUM_Q_HEADS,
        scale_value=MLA_SOFTMAX_SCALE,
        block_table=block_table.to(DEVICE),
        kv_seq_len=torch.tensor(kv_seq_lens, dtype=torch.int32),
        mla_vheadsize=MLA_V_HEAD_DIM,
        attn_output=output,
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=5e-3, atol=5e-3)
