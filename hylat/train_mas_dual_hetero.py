# Modified from train_decoder_mas_dual.py to support heterogeneous agents.
# Agent1 and Agent2 use different backbone models (e.g. Llama + Qwen).
# Each agent's data is tokenized with its own tokenizer.
# Two bidirectional adapters (adapter_1to2, adapter_2to1) are trained jointly.
import copy
import logging
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import json
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from safetensors.torch import load_file
from tqdm import tqdm
from math import ceil
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from functools import partial
import argparse

# Import chat-template helpers and misc utils from train.py
from train import (
    add_chat_template_prefix_qwen,
    add_chat_template_suffix_qwen,
    add_chat_template_prefix_llama,
    add_chat_template_suffix_llama,
    add_chat_template_prefix_gemma,
    add_chat_template_suffix_gemma,
    add_chat_template_prefix_ministral,
    add_chat_template_suffix_ministral,
    add_chat_template_prefix_olmo,
    add_chat_template_suffix_olmo,
)

from src.model_decoder_mas_dual_hetero import (
    HyLAT,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model,
)

IGNORE_INDEX = -100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# ── Chat-template dispatch (module-level, works for both agents) ──────────────

def add_chat_template_prefix(text: str, add_system: bool, model_name: str) -> str:
    n = model_name.lower()
    if "qwen" in n:
        return add_chat_template_prefix_qwen(text, add_system)
    elif "llama" in n:
        return add_chat_template_prefix_llama(text, add_system)
    elif "gemma" in n:
        return add_chat_template_prefix_gemma(text, add_system)
    elif "ministral" in n:
        return add_chat_template_prefix_ministral(text, add_system)
    elif "olmo" in n:
        return add_chat_template_prefix_olmo(text, add_system)
    else:
        raise NotImplementedError(f"No chat template for: {model_name}")


def add_chat_template_suffix(text: str, model_name: str) -> str:
    n = model_name.lower()
    if "qwen" in n:
        return add_chat_template_suffix_qwen(text)
    elif "llama" in n:
        return add_chat_template_suffix_llama(text)
    elif "gemma" in n:
        return add_chat_template_suffix_gemma(text)
    elif "ministral" in n:
        return add_chat_template_suffix_ministral(text)
    elif "olmo" in n:
        return add_chat_template_suffix_olmo(text)
    else:
        raise NotImplementedError(f"No chat template for: {model_name}")


# ── Lora config helper ────────────────────────────────────────────────────────

def make_lora_config(model_name: str, r: int, alpha: int, dropout: float) -> LoraConfig:
    n = model_name.lower()
    if any(x in n for x in ["llama", "mistral", "falcon", "qwen", "olmo", "gemma", "ministral"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "phi" in n:
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif "gpt2" in n:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model for LoRA target modules: {model_name}")
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        init_lora_weights=True,
    )


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_short_answer(text: str) -> str:
    text = text.strip()
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()
    answer_is = re.findall(r'[Tt]he answer is[:\s]+([^\n.]+)', text)
    if answer_is:
        candidate = answer_is[-1].strip().rstrip('.')
        if len(candidate) <= 50:
            return candidate
    letter = re.search(r'\(([A-E])\)|(?<!\w)([A-E])(?!\w)', text)
    if letter:
        return (letter.group(1) or letter.group(2)).strip()
    if '####' in text:
        val = text.split('####')[-1].strip().replace(',', '')
        num = re.search(r'-?\d+\.?\d*', val)
        if num:
            return num.group()
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        return nums[-1]
    return text


def get_answer_token_position(tokens: torch.Tensor, answer_prompts: list) -> int:
    try:
        match_indices = (
            tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]
        ).all(dim=1).nonzero(as_tuple=True)[0].item()
        return match_indices + len(answer_prompts[0])
    except Exception:
        # Fallback to second prompt pattern
        try:
            match_indices = (
                tokens.unfold(0, len(answer_prompts[1]), 1) == answer_prompts[1]
            ).all(dim=1).nonzero(as_tuple=True)[0].item()
            return match_indices + len(answer_prompts[1])
        except Exception:
            return 0


# ── Context builder (called with the READING agent's tokenizer/model_name) ────

def build_agent_context(
    dialogue: dict,
    tokenizer,
    bot_token: str,
    eot_token: str,
    latent_token: str,
    num_latent: int,
    model_name: str,
):
    """Build the context that an agent reads from the other agent's previous response."""
    OTHER_TEMPLATE = [
        "These are the solutions to the problem from other agents: {content}\n\n"
        "Using the responses from other agents as additional information, can you give an answer "
        "to the original question?\nExamine your solution and that other agents step by step.\n"
        "Make sure to state your answer at the end of the response.",
        "These are the responses from other agents: {content}\n\n"
        "What is your answer to the original question? "
        "Make sure to state your answer at the end of the response.",
    ]
    other_answer = dialogue["answer"]
    other_cot    = dialogue["cot"]
    other_latent = latent_token * num_latent

    template = random.choice(OTHER_TEMPLATE)

    # Latent-based context
    response = bot_token + other_latent + eot_token + other_answer
    context  = add_chat_template_prefix(template.format(content=response),
                                        add_system=False, model_name=model_name)
    # Text-only context (for teacher task reference)
    text_ctx = add_chat_template_prefix(
        template.format(content=f"{other_cot} {other_answer}"),
        add_system=False, model_name=model_name,
    )
    return (
        tokenizer.encode(context, add_special_tokens=False),
        tokenizer.encode(text_ctx, add_special_tokens=False),
    )


# ── Data preprocessing ────────────────────────────────────────────────────────

def preprocess_mas(
    dialogue_list: Sequence,
    # Agent1
    tokenizer1,
    bot_id1: int, eot_id1: int, latent_id1: int,
    bot_token1: str, eot_token1: str,
    # Agent2
    tokenizer2,
    bot_id2: int, eot_id2: int, latent_id2: int,
    bot_token2: str, eot_token2: str,
    # Shared
    latent_token: str,
    num_latent: int,
    model_name1: str,
    model_name2: str,
    mask_inter_res: bool,
) -> Dict:
    """
    Tokenize dialogues for heterogeneous agents.
    Agent1's sequences use tokenizer1 / model_name1.
    Agent2's sequences use tokenizer2 / model_name2.
    """
    print("Tokenizing inputs… this may take some time.")

    # Per-tokenizer answer prompt tensors (used to find answer start position)
    def make_answer_prompts(tok):
        prompts = [
            torch.tensor(tok.encode("The answer is:")),
            torch.tensor(tok.encode("I would say:")),
        ]
        if prompts[0][0] == tok.bos_token_id:
            prompts = [p[1:] for p in prompts]
        return prompts

    answer_prompts1 = make_answer_prompts(tokenizer1)
    answer_prompts2 = make_answer_prompts(tokenizer2)

    # Output lists
    (a1_enc_list, a1_dec_list, a1_lbl_list,
     a1_model_pos_list, a1_ref_pos_list,
     a1_full_ref_list, a1_full_ref_lbl_list,
     a1_long_dec_list, a1_long_lbl_list) = ([] for _ in range(9))

    (a2_enc_list, a2_dec_list, a2_lbl_list,
     a2_model_pos_list, a2_ref_pos_list,
     a2_full_ref_list, a2_full_ref_lbl_list,
     a2_long_dec_list, a2_long_lbl_list) = ([] for _ in range(9))

    mask_inter_res_list = []

    for dialogue_turns in dialogue_list:
        sample_mask = dialogue_turns[0].get("mask_inter_res", mask_inter_res)

        # Per-dialog accumulators
        (a1_enc, a1_dec, a1_lbl, a1_mpos, a1_rpos,
         a1_long_dec, a1_long_lbl, a1_full_ref_toks) = ([] for _ in range(8))
        a1_cum_len = 0
        a1_spans = []

        (a2_enc, a2_dec, a2_lbl, a2_mpos, a2_rpos,
         a2_long_dec, a2_long_lbl, a2_full_ref_toks) = ([] for _ in range(8))
        a2_cum_len = 0
        a2_spans = []

        init_q1 = dialogue_turns[0].get("question1", None)
        init_q2 = dialogue_turns[0].get("question2", None)
        turns = dialogue_turns[1:]

        for turn_idx, turn_data in enumerate(turns):
            cot    = turn_data["cot"]
            answer = turn_data["answer"]

            # ── Agent1 (even turns) ──────────────────────────────────────────
            if turn_idx % 2 == 0:
                if turn_idx == 0:
                    q_text = add_chat_template_prefix(init_q1, add_system=True, model_name=model_name1)
                    q_toks = tokenizer1.encode(q_text, add_special_tokens=False)
                    ctx_toks, txt_ctx_toks = [], []
                else:
                    ctx_toks, txt_ctx_toks = build_agent_context(
                        turns[turn_idx - 1], tokenizer1,
                        bot_token1, eot_token1, latent_token, num_latent, model_name1,
                    )
                    q_text, q_toks = "", []

                enc_toks = ctx_toks + q_toks + [bot_id1]
                a1_enc.append(torch.tensor(enc_toks, dtype=torch.long))

                ans_text = add_chat_template_suffix(answer, model_name1)
                ans_toks = tokenizer1.encode(ans_text, add_special_tokens=False)
                dec_ids  = torch.tensor([eot_id1] + ans_toks, dtype=torch.long)
                a1_dec.append(dec_ids)
                a1_lbl.append(dec_ids.clone())
                a1_mpos.append(get_answer_token_position(dec_ids, answer_prompts1))

                if turn_idx == 0:
                    ref_text = q_text + cot + ans_text
                else:
                    ref_text = cot + ans_text
                ref_toks = txt_ctx_toks + tokenizer1.encode(ref_text, add_special_tokens=False)
                ref_tensor = torch.tensor(ref_toks, dtype=torch.long)
                local_ref_pos  = get_answer_token_position(ref_tensor, answer_prompts1)
                global_ref_pos = a1_cum_len + local_ref_pos
                a1_rpos.append(global_ref_pos)

                ans_start = a1_cum_len + len(ref_toks) - len(ans_toks)
                ans_end   = a1_cum_len + len(ref_toks)
                a1_spans.append((ans_start, ans_end))
                a1_cum_len += len(ref_toks)
                a1_full_ref_toks.extend(ref_toks)

                cot_toks = tokenizer1.encode(cot, add_special_tokens=False)
                long_dec = torch.tensor(cot_toks, dtype=torch.long)
                long_lbl = torch.cat([
                    torch.full((num_latent + 1,), -100, dtype=torch.long),
                    long_dec,
                ])
                a1_long_dec.append(long_dec)
                a1_long_lbl.append(long_lbl)

            # ── Agent2 (odd turns) ───────────────────────────────────────────
            if turn_idx % 2 == 1:
                if turn_idx == 1:
                    q_text = add_chat_template_prefix(init_q2, add_system=True, model_name=model_name2)
                    q_toks = tokenizer2.encode(q_text, add_special_tokens=False)
                    ctx_toks, txt_ctx_toks = [], []
                else:
                    ctx_toks, txt_ctx_toks = build_agent_context(
                        turns[turn_idx - 1], tokenizer2,
                        bot_token2, eot_token2, latent_token, num_latent, model_name2,
                    )
                    q_text, q_toks = "", []

                enc_toks = ctx_toks + q_toks + [bot_id2]
                a2_enc.append(torch.tensor(enc_toks, dtype=torch.long))

                ans_text = add_chat_template_suffix(answer, model_name2)
                ans_toks = tokenizer2.encode(ans_text, add_special_tokens=False)
                dec_ids  = torch.tensor([eot_id2] + ans_toks, dtype=torch.long)
                a2_dec.append(dec_ids)
                a2_lbl.append(dec_ids.clone())
                a2_mpos.append(get_answer_token_position(dec_ids, answer_prompts2))

                if turn_idx == 1:
                    ref_text = q_text + cot + ans_text
                else:
                    ref_text = cot + ans_text
                ref_toks = txt_ctx_toks + tokenizer2.encode(ref_text, add_special_tokens=False)
                ref_tensor = torch.tensor(ref_toks, dtype=torch.long)
                local_ref_pos  = get_answer_token_position(ref_tensor, answer_prompts2)
                global_ref_pos = a2_cum_len + local_ref_pos
                a2_rpos.append(global_ref_pos)

                ans_start = a2_cum_len + len(ref_toks) - len(ans_toks)
                ans_end   = a2_cum_len + len(ref_toks)
                a2_spans.append((ans_start, ans_end))
                a2_cum_len += len(ref_toks)
                a2_full_ref_toks.extend(ref_toks)

                cot_toks = tokenizer2.encode(cot, add_special_tokens=False)
                long_dec = torch.tensor(cot_toks, dtype=torch.long)
                long_lbl = torch.cat([
                    torch.full((num_latent + 1,), -100, dtype=torch.long),
                    long_dec,
                ])
                a2_long_dec.append(long_dec)
                a2_long_lbl.append(long_lbl)

        # ── Assemble full_ref tensors with label masking ─────────────────────
        a1_full_ref_ids = torch.tensor(a1_full_ref_toks, dtype=torch.long)
        a1_full_ref_lbls = torch.full_like(a1_full_ref_ids, -100)
        for start, end in (a1_spans[-1:] if sample_mask else a1_spans):
            a1_full_ref_lbls[start:end] = a1_full_ref_ids[start:end]

        a2_full_ref_ids = torch.tensor(a2_full_ref_toks, dtype=torch.long)
        a2_full_ref_lbls = torch.full_like(a2_full_ref_ids, -100)
        for start, end in (a2_spans[-1:] if sample_mask else a2_spans):
            a2_full_ref_lbls[start:end] = a2_full_ref_ids[start:end]

        a1_enc_list.append(a1_enc);     a1_dec_list.append(a1_dec)
        a1_lbl_list.append(a1_lbl);     a1_model_pos_list.append(a1_mpos)
        a1_ref_pos_list.append(a1_rpos);a1_full_ref_list.append(a1_full_ref_ids)
        a1_full_ref_lbl_list.append(a1_full_ref_lbls)
        a1_long_dec_list.append(a1_long_dec); a1_long_lbl_list.append(a1_long_lbl)

        a2_enc_list.append(a2_enc);     a2_dec_list.append(a2_dec)
        a2_lbl_list.append(a2_lbl);     a2_model_pos_list.append(a2_mpos)
        a2_ref_pos_list.append(a2_rpos);a2_full_ref_list.append(a2_full_ref_ids)
        a2_full_ref_lbl_list.append(a2_full_ref_lbls)
        a2_long_dec_list.append(a2_long_dec); a2_long_lbl_list.append(a2_long_lbl)

        mask_inter_res_list.append(torch.tensor(sample_mask, dtype=torch.bool))

    return dict(
        agent1_encoder_input_ids_list=a1_enc_list,
        agent1_decoder_input_ids_list=a1_dec_list,
        agent1_labels_list=a1_lbl_list,
        agent1_model_answer_position=a1_model_pos_list,
        agent1_ref_answer_position=a1_ref_pos_list,
        agent1_full_ref_input_ids=a1_full_ref_list,
        agent1_full_ref_labels=a1_full_ref_lbl_list,
        agent1_long_decoder_input_ids_list=a1_long_dec_list,
        agent1_long_labels_list=a1_long_lbl_list,

        agent2_encoder_input_ids_list=a2_enc_list,
        agent2_decoder_input_ids_list=a2_dec_list,
        agent2_labels_list=a2_lbl_list,
        agent2_model_answer_position=a2_model_pos_list,
        agent2_ref_answer_position=a2_ref_pos_list,
        agent2_full_ref_input_ids=a2_full_ref_list,
        agent2_full_ref_labels=a2_full_ref_lbl_list,
        agent2_long_decoder_input_ids_list=a2_long_dec_list,
        agent2_long_labels_list=a2_long_lbl_list,

        mask_inter_res=mask_inter_res_list,
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class MASSupervisedDataset(Dataset):
    def __init__(
        self,
        data_name, raw_data,
        tokenizer1, tokenizer2,
        bot_id1, eot_id1, latent_id1, bot_token1, eot_token1,
        bot_id2, eot_id2, latent_id2, bot_token2, eot_token2,
        latent_token, num_latent,
        model_name1, model_name2,
        training_args,
        fixed_num_turns=2,
        mask_inter_res=False,
    ):
        super().__init__()
        logging.warning("Formatting inputs…")

        dialogue_list = []
        for num_iter, example in enumerate(raw_data):
            if training_args.exp_mode and num_iter > training_args.exp_data_num:
                break

            messages = example["messages"][:1 + 2 * fixed_num_turns]
            if len(messages) < 2 * fixed_num_turns:
                continue

            turns = []
            for i, msg in enumerate(messages):
                q1 = msg.get("question_agent1", "") if i == 0 else ""
                q2 = msg.get("question_agent2", "") if i == 0 else ""
                short_answer = str(msg["short_answer"])
                long_answer  = str(msg["long_answer"])
                cot = long_answer.split(". ")
                if not training_args.include_last_cot:
                    cot = cot[:-1]
                cot = (". ".join(cot) + ".\n") if cot else ""
                answer = extract_short_answer(short_answer)
                answer = f"The answer is: {answer}".replace("####", "")
                turns.append({
                    "question1": q1, "question2": q2,
                    "cot": cot, "answer": answer,
                    "mask_inter_res": example.get("mask_inter_res", mask_inter_res),
                })

            # Filter by length using tokenizer1 as reference
            token_num = len(tokenizer1.encode(
                " ".join(t["question1"] + t["cot"] + t["answer"] for t in turns)
            ))
            if token_num > training_args.max_token_num:
                continue
            if turns:
                dialogue_list.append(turns)

        if training_args.exp_mode:
            dialogue_list = dialogue_list[:training_args.exp_data_num]

        print(f"{len(dialogue_list)} dialogues, "
              f"{sum(len(d) for d in dialogue_list)} turns total.")

        self.data_dict = preprocess_mas(
            dialogue_list,
            tokenizer1, bot_id1, eot_id1, latent_id1, bot_token1, eot_token1,
            tokenizer2, bot_id2, eot_id2, latent_id2, bot_token2, eot_token2,
            latent_token, num_latent,
            model_name1, model_name2,
            mask_inter_res,
        )
        self.keys = list(self.data_dict.keys())

    def __len__(self):
        return len(self.data_dict["agent1_full_ref_input_ids"])

    def __getitem__(self, i):
        return {k: self.data_dict[k][i] for k in self.keys}


# ── Collator ──────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorForSupervisedDataset:
    """
    Pads agent1 sequences with pad_token_id1 and agent2 sequences with pad_token_id2.
    """
    pad_token_id1: int
    pad_token_id2: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_turns  = max(len(inst["agent1_encoder_input_ids_list"]) for inst in instances)
        batch_size = len(instances)

        def pad_turn_list(per_inst_list, max_t, pad_val, fill_val=None):
            """Return list-of-lists padded to max_t turns."""
            pad_tensor = (
                torch.zeros(1, dtype=torch.long) if fill_val is None
                else torch.full((1,), fill_val, dtype=torch.long)
            )
            return [
                inst + [pad_tensor] * (max_t - len(inst))
                for inst in per_inst_list
            ]

        def stack_turns(
            per_inst_enc, per_inst_dec, per_inst_lbl,
            per_inst_long_dec, per_inst_long_lbl,
            pad_id,
        ):
            """Pad and stack per-turn tensors into [B, T, L]."""
            # Find max lengths across all turns and samples
            max_enc = max_dec = max_long_dec = max_long_lbl = 0
            for ti in range(max_turns):
                max_enc      = max(max_enc, max(inst[ti].size(0) for inst in per_inst_enc))
                max_dec      = max(max_dec, max(inst[ti].size(0) for inst in per_inst_dec))
                max_long_dec = max(max_long_dec, max(inst[ti].size(0) for inst in per_inst_long_dec))
                max_long_lbl = max(max_long_lbl, max(inst[ti].size(0) for inst in per_inst_long_lbl))

            enc_final = []; dec_final = []; lbl_final = []
            long_dec_final = []; long_lbl_final = []

            for ti in range(max_turns):
                turn_enc      = [inst[ti] for inst in per_inst_enc]
                turn_dec      = [inst[ti] for inst in per_inst_dec]
                turn_lbl      = [inst[ti] for inst in per_inst_lbl]
                turn_long_dec = [inst[ti] for inst in per_inst_long_dec]
                turn_long_lbl = [inst[ti] for inst in per_inst_long_lbl]

                # Left-pad encoder
                rev_enc = [s.flip(0) for s in turn_enc]
                padded_enc = torch.nn.utils.rnn.pad_sequence(
                    rev_enc, batch_first=True, padding_value=pad_id).flip(1)
                if padded_enc.size(1) < max_enc:
                    padded_enc = torch.cat([
                        torch.full((batch_size, max_enc - padded_enc.size(1)), pad_id, dtype=torch.long),
                        padded_enc,
                    ], dim=1)

                # Right-pad decoder
                padded_dec = torch.nn.utils.rnn.pad_sequence(
                    turn_dec, batch_first=True, padding_value=pad_id)
                if padded_dec.size(1) < max_dec:
                    padded_dec = torch.cat([
                        padded_dec,
                        torch.full((batch_size, max_dec - padded_dec.size(1)), pad_id, dtype=torch.long),
                    ], dim=1)

                padded_lbl = torch.nn.utils.rnn.pad_sequence(
                    turn_lbl, batch_first=True, padding_value=IGNORE_INDEX)
                if padded_lbl.size(1) < max_dec:
                    padded_lbl = torch.cat([
                        padded_lbl,
                        torch.full((batch_size, max_dec - padded_lbl.size(1)), IGNORE_INDEX, dtype=torch.long),
                    ], dim=1)

                padded_long_dec = torch.nn.utils.rnn.pad_sequence(
                    turn_long_dec, batch_first=True, padding_value=pad_id)
                if padded_long_dec.size(1) < max_long_dec:
                    padded_long_dec = torch.cat([
                        padded_long_dec,
                        torch.full((batch_size, max_long_dec - padded_long_dec.size(1)), pad_id, dtype=torch.long),
                    ], dim=1)

                padded_long_lbl = torch.nn.utils.rnn.pad_sequence(
                    turn_long_lbl, batch_first=True, padding_value=IGNORE_INDEX)
                if padded_long_lbl.size(1) < max_long_lbl:
                    padded_long_lbl = torch.cat([
                        padded_long_lbl,
                        torch.full((batch_size, max_long_lbl - padded_long_lbl.size(1)), IGNORE_INDEX, dtype=torch.long),
                    ], dim=1)

                enc_final.append(padded_enc);         dec_final.append(padded_dec)
                lbl_final.append(padded_lbl);         long_dec_final.append(padded_long_dec)
                long_lbl_final.append(padded_long_lbl)

            # [T, B, L] -> [B, T, L]
            return (
                torch.stack(enc_final).transpose(0, 1),
                torch.stack(dec_final).transpose(0, 1),
                torch.stack(lbl_final).transpose(0, 1),
                torch.stack(long_dec_final).transpose(0, 1),
                torch.stack(long_lbl_final).transpose(0, 1),
            )

        # ── Agent1 ──────────────────────────────────────────────────────────
        a1_enc_batch  = pad_turn_list([i["agent1_encoder_input_ids_list"] for i in instances], max_turns, self.pad_token_id1)
        a1_dec_batch  = pad_turn_list([i["agent1_decoder_input_ids_list"] for i in instances], max_turns, self.pad_token_id1)
        a1_lbl_batch  = pad_turn_list([i["agent1_labels_list"] for i in instances], max_turns, self.pad_token_id1, fill_val=IGNORE_INDEX)
        a1_ldec_batch = pad_turn_list([i["agent1_long_decoder_input_ids_list"] for i in instances], max_turns, self.pad_token_id1)
        a1_llbl_batch = pad_turn_list([i["agent1_long_labels_list"] for i in instances], max_turns, self.pad_token_id1, fill_val=IGNORE_INDEX)

        a1_enc_stk, a1_dec_stk, a1_lbl_stk, a1_ldec_stk, a1_llbl_stk = stack_turns(
            a1_enc_batch, a1_dec_batch, a1_lbl_batch, a1_ldec_batch, a1_llbl_batch, self.pad_token_id1)

        a1_full_ref_ids = torch.nn.utils.rnn.pad_sequence(
            [i["agent1_full_ref_input_ids"] for i in instances], batch_first=True, padding_value=self.pad_token_id1)
        a1_full_ref_lbl = torch.nn.utils.rnn.pad_sequence(
            [i["agent1_full_ref_labels"] for i in instances], batch_first=True, padding_value=IGNORE_INDEX)
        a1_model_pos = torch.tensor([i["agent1_model_answer_position"] for i in instances], dtype=torch.long)
        a1_ref_pos   = torch.tensor([i["agent1_ref_answer_position"] for i in instances], dtype=torch.long)

        # ── Agent2 ──────────────────────────────────────────────────────────
        a2_enc_batch  = pad_turn_list([i["agent2_encoder_input_ids_list"] for i in instances], max_turns, self.pad_token_id2)
        a2_dec_batch  = pad_turn_list([i["agent2_decoder_input_ids_list"] for i in instances], max_turns, self.pad_token_id2)
        a2_lbl_batch  = pad_turn_list([i["agent2_labels_list"] for i in instances], max_turns, self.pad_token_id2, fill_val=IGNORE_INDEX)
        a2_ldec_batch = pad_turn_list([i["agent2_long_decoder_input_ids_list"] for i in instances], max_turns, self.pad_token_id2)
        a2_llbl_batch = pad_turn_list([i["agent2_long_labels_list"] for i in instances], max_turns, self.pad_token_id2, fill_val=IGNORE_INDEX)

        a2_enc_stk, a2_dec_stk, a2_lbl_stk, a2_ldec_stk, a2_llbl_stk = stack_turns(
            a2_enc_batch, a2_dec_batch, a2_lbl_batch, a2_ldec_batch, a2_llbl_batch, self.pad_token_id2)

        a2_full_ref_ids = torch.nn.utils.rnn.pad_sequence(
            [i["agent2_full_ref_input_ids"] for i in instances], batch_first=True, padding_value=self.pad_token_id2)
        a2_full_ref_lbl = torch.nn.utils.rnn.pad_sequence(
            [i["agent2_full_ref_labels"] for i in instances], batch_first=True, padding_value=IGNORE_INDEX)
        a2_model_pos = torch.tensor([i["agent2_model_answer_position"] for i in instances], dtype=torch.long)
        a2_ref_pos   = torch.tensor([i["agent2_ref_answer_position"] for i in instances], dtype=torch.long)

        mask_inter_res_batch = torch.stack([inst["mask_inter_res"] for inst in instances])

        return dict(
            agent1_encoder_input_ids_list=a1_enc_stk,
            agent1_decoder_input_ids_list=a1_dec_stk,
            agent1_labels_list=a1_lbl_stk,
            agent1_encoder_attention_mask_list=a1_enc_stk.ne(self.pad_token_id1),
            agent1_model_answer_position=a1_model_pos,
            agent1_ref_answer_position=a1_ref_pos,
            agent1_full_ref_input_ids=a1_full_ref_ids,
            agent1_full_ref_labels=a1_full_ref_lbl,
            agent1_full_ref_attention_mask=a1_full_ref_ids.ne(self.pad_token_id1),
            agent1_long_decoder_input_ids_list=a1_ldec_stk,
            agent1_long_labels_list=a1_llbl_stk,

            agent2_encoder_input_ids_list=a2_enc_stk,
            agent2_decoder_input_ids_list=a2_dec_stk,
            agent2_labels_list=a2_lbl_stk,
            agent2_encoder_attention_mask_list=a2_enc_stk.ne(self.pad_token_id2),
            agent2_model_answer_position=a2_model_pos,
            agent2_ref_answer_position=a2_ref_pos,
            agent2_full_ref_input_ids=a2_full_ref_ids,
            agent2_full_ref_labels=a2_full_ref_lbl,
            agent2_full_ref_attention_mask=a2_full_ref_ids.ne(self.pad_token_id2),
            agent2_long_decoder_input_ids_list=a2_ldec_stk,
            agent2_long_labels_list=a2_llbl_stk,

            mask_inter_res=mask_inter_res_batch,
        )


# ── Trainer ───────────────────────────────────────────────────────────────────

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step
        batch_size = self.args.per_device_train_batch_size
        effective_bs = batch_size * self.args.world_size * self.args.gradient_accumulation_steps
        total_steps = ceil(len(self.train_dataset) / effective_bs) * self.args.num_train_epochs

        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        outputs = model(**inputs)
        loss = outputs["loss"]

        if step % self.args.logging_steps == 0:
            self.log({
                "loss": loss.item(),
                "ce_loss": outputs["ce_loss"],
                "distill_loss": outputs["distill_loss"],
                "ref_ce_loss": outputs["ref_ce_loss"],
                "decode_loss": outputs["decode_loss"],
                "loss_detail": outputs["loss_detail"],
            })
        return loss

    def create_optimizer(self):
        """Split parameters into two groups: LLM (codi1/codi2/prj) and adapter."""
        adapter_lr = (
            float(self.args.adapter_lr)
            if self.args.adapter_lr > 0
            else float(self.args.learning_rate)
        )
        llm_lr = float(self.args.learning_rate)

        adapter_params, llm_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "adapter_1to2" in name or "adapter_2to1" in name:
                adapter_params.append(param)
            else:
                llm_params.append(param)

        param_groups = []
        if llm_params:
            param_groups.append({"params": llm_params, "lr": llm_lr})
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": adapter_lr})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=llm_lr,
            weight_decay=self.args.weight_decay,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(args.conf, allow_extra_keys=True)

    # Build LoRA configs (one per model)
    lora_config1 = lora_config2 = None
    if model_args.lora_init:
        lora_config1 = make_lora_config(
            model_args.model_name_or_path, model_args.lora_r,
            model_args.lora_alpha, model_args.lora_dropout,
        )
        lora_config2 = make_lora_config(
            model_args.model2_name_or_path, model_args.lora_r2,
            model_args.lora_alpha2, model_args.lora_dropout,
        )

    model = HyLAT(model_args, training_args, lora_config1, lora_config2)

    # ── Load stage1 checkpoints (remap codi.→codi1. / codi.→codi2.) ─────────
    def _load_stage1(ckpt_dir, model_prefix, prj_prefix, tie_model):
        try:
            sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        except Exception:
            sd = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
        remapped = {}
        for k, v in sd.items():
            if k.startswith("codi."):
                remapped[model_prefix + k[5:]] = v
            elif k.startswith("prj."):
                remapped[prj_prefix + k[4:]] = v
        # Keep freshly-initialized embeddings to avoid vocab-size mismatch
        cur_sd = model.state_dict()
        for embed_key in [f"{model_prefix}base_model.model.model.embed_tokens.weight",
                          f"{model_prefix}base_model.model.lm_head.weight"]:
            if embed_key in remapped and embed_key in cur_sd:
                remapped[embed_key] = cur_sd[embed_key]
        model.load_state_dict(remapped, strict=False)
        tie_model.tie_weights()

    if model_args.stage1_ckpt_dir:
        _load_stage1(model_args.stage1_ckpt_dir, "codi1.", "prj1.", model.codi1)
    if model_args.stage1_ckpt_dir2:
        _load_stage1(model_args.stage1_ckpt_dir2, "codi2.", "prj2.", model.codi2)

    # Tokenizer setup: fix pad_token_id if needed
    tokenizer1 = model.tokenizer1
    tokenizer2 = model.tokenizer2
    if tokenizer1.pad_token_id is None:
        tokenizer1.pad_token_id = model.pad_token_id1
    if tokenizer2.pad_token_id is None:
        tokenizer2.pad_token_id = model.pad_token_id2

    model_name1 = model_args.model_name_or_path.split("/")[-1].lower()
    model_name2 = model_args.model2_name_or_path.split("/")[-1].lower()

    def make_supervised_data_module(tokenizer1, tokenizer2, data_args, model_args, training_args):
        dataset = load_dataset("json", data_files=data_args.data_name)["train"]
        train_dataset = MASSupervisedDataset(
            data_name=data_args.data_name,
            raw_data=dataset,
            tokenizer1=tokenizer1,
            tokenizer2=tokenizer2,
            bot_id1=model.bot_id1, eot_id1=model.eot_id1, latent_id1=model.latent_id1,
            bot_token1=model.bot_token1, eot_token1=model.eot_token1,
            bot_id2=model.bot_id2, eot_id2=model.eot_id2, latent_id2=model.latent_id2,
            bot_token2=model.bot_token2, eot_token2=model.eot_token2,
            latent_token=model.latent_token,
            num_latent=model.num_latent,
            model_name1=model_name1,
            model_name2=model_name2,
            training_args=training_args,
            fixed_num_turns=data_args.fixed_num_turns,
            mask_inter_res=training_args.mask_inter_res,
        )
        data_collator = DataCollatorForSupervisedDataset(
            pad_token_id1=model.pad_token_id1,
            pad_token_id2=model.pad_token_id2,
        )
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        f"{model_name1}_x_{model_name2}",
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(tokenizer1, tokenizer2, data_args, model_args, training_args)
    training_args.learning_rate = float(training_args.learning_rate)
    training_args.adapter_lr = float(training_args.adapter_lr)
    

    # Pass tokenizer1 as the primary tokenizer for HF Trainer bookkeeping
    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer1, args=training_args, **data_module
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf", type=str,
        default="configs/stage2_hetero_llama_qwen.yaml",
    )
    args = parser.parse_args()
    train(args)
