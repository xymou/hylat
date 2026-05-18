import torch
from typing import Callable, List, Dict, Tuple
from transformers import AutoTokenizer

HIDDEN_DIM = None

def update_hidden_dim(hidden_dim):
    global HIDDEN_DIM
    HIDDEN_DIM = hidden_dim


def get_merged_embedding_for_cipher(
    t2e_func: Callable,
    prompt_template: str,
    placeholder: str,
    input_embs,
):
    start_pos = prompt_template.find(placeholder)
    prefix = prompt_template[:start_pos]
    suffix = prompt_template[start_pos + len(placeholder):]
    merged = [
        t2e_func(prefix), 
        input_embs,
        t2e_func(suffix),
    ]
    return torch.cat(merged, dim=0)

def get_merged_ids_mask_hs_for_sde(
    tokenizer: AutoTokenizer,
    prompt_template: str,
    placeholder: str,
    input_ids: List[int],
    input_mask: torch.Tensor,
    input_hs: Dict[int, torch.Tensor], 
) -> Tuple[List, torch.BoolTensor, Dict[int, torch.Tensor]]:
    start_pos = prompt_template.find(placeholder)
    prefix = prompt_template[:start_pos]
    suffix = prompt_template[start_pos + len(placeholder):]

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    merged_ids = prefix_ids + input_ids + suffix_ids

    merged_mask = torch.ones(len(merged_ids), dtype=input_mask.dtype)
    # 中间 input_ids 部分换成 input_mask
    merged_mask[len(prefix_ids):len(prefix_ids)+len(input_ids)] = input_mask

    merged_hs = {}
    for layer_idx, hs in input_hs.items():
        if hs is None:
            assert len(input_ids) == 0
            merged_hs[layer_idx] = torch.zeros(1, len(merged_ids), HIDDEN_DIM)
        else:
            merged_hs[layer_idx] = torch.cat([
                torch.zeros(1, len(prefix_ids), hs.shape[-1], dtype=hs.dtype),
                hs, 
                torch.zeros(1, len(suffix_ids), hs.shape[-1], dtype=hs.dtype),
            ], dim=1)
    return merged_ids, merged_mask, merged_hs


def get_merged_embedding_for_latent(
    t2e_func: Callable,
    prompt_template: str,
    placeholder: str,
    input_embs,
):
    start_pos = prompt_template.find(placeholder)
    prefix = prompt_template[:start_pos]
    suffix = prompt_template[start_pos + len(placeholder):]
    if prefix and suffix:
        merged = [
            t2e_func(prefix), 
            input_embs,
            t2e_func(suffix),
        ]
        return torch.cat(merged, dim=0)
    return input_embs

