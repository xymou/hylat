# Generic N-agent MAS training script.
# Generalizes train_decoder_mas_dual.py to support any num_agents >= 2.
#
# Data format expected in each example["messages"]:
#   messages[0]: metadata with keys question_agent1...question_agent{N},
#                short_answer, long_answer (unused for turn0 CoT/answer)
#   messages[1..N]:   agent0..agent{N-1} round0 responses
#   messages[N+1..2N]: agent0..agent{N-1} round1 responses
#   ...  (total: 1 + num_agents * fixed_num_turns messages)
#
# Agent keys in batch are 0-indexed: agent0_*, agent1_*, ...
import copy
import logging
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
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
import argparse
from train import *

from src.model_decoder_mas_general import (
    HyLAT,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model,
)

IGNORE_INDEX = -100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def extract_short_answer(text):
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


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step
        batch_size = self.args.per_device_train_batch_size
        effective_batch = batch_size * self.args.world_size * self.args.gradient_accumulation_steps
        total_steps = ceil(len(self.train_dataset) / effective_batch) * self.args.num_train_epochs
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

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256,
            truncation=True,
            return_attention_mask=False,
        )
        for text in strings
    ]
    input_ids = [t.input_ids[0] for t in tokenized_list]
    input_ids_lens = [ids.ne(tokenizer.pad_token_id).sum().item() for ids in input_ids]
    return dict(input_ids=input_ids, labels=input_ids, input_ids_lens=input_ids_lens, labels_lens=input_ids_lens)


def train(args):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(args.conf, allow_extra_keys=True)

    num_agents = data_args.num_agents

    # ── LoRA config ────────────────────────────────────────────────────────────
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        name = model_args.model_name_or_path.lower()
        if any(n in name for n in ["llama", "mistral", "falcon", "qwen", "olmo"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif "phi" in name:
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif "gpt2" in name:
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )

    model = HyLAT(model_args, training_args, lora_config)

    if model_args.stage1_ckpt_dir:
        try:
            state_dict = load_file(os.path.join(model_args.stage1_ckpt_dir, "model.safetensors"))
        except Exception:
            state_dict = torch.load(os.path.join(model_args.stage1_ckpt_dir, "pytorch_model.bin"))
        model_state = model.state_dict()
        embed_key = [
            "codi.base_model.model.model.embed_tokens.weight",
            "codi.base_model.model.lm_head.weight",
        ]
        for key in embed_key:
            state_dict[key] = model_state[key]
        model.load_state_dict(state_dict, strict=False)
        model.codi.tie_weights()

    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    # ── Helper: answer token position ─────────────────────────────────────────
    def get_answer_token_position(tokens, answer_prompts, tokenizer):
        try:
            match = (
                tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]
            ).all(dim=1).nonzero(as_tuple=True)[0].item()
            return match + len(answer_prompts[0])
        except Exception:
            breakpoint()

    # ── Chat template wrappers (delegates to functions imported from train.py) ─
    model_name_lower = model_args.model_name_or_path.split('/')[-1].lower()

    def add_chat_template_prefix(text, add_system):
        return add_chat_template_prefix.__wrapped__(text, add_system, model_name_lower)

    # Resolve the real function from train.py imports at call time
    def _prefix(text, add_system, name):
        if 'qwen' in name:
            return add_chat_template_prefix_qwen(text, add_system)
        elif 'llama' in name:
            return add_chat_template_prefix_llama(text, add_system)
        elif 'gemma' in name:
            return add_chat_template_prefix_gemma(text, add_system)
        elif 'ministral' in name:
            return add_chat_template_prefix_ministral(text, add_system)
        elif 'olmo' in name:
            return add_chat_template_prefix_olmo(text, add_system)
        raise NotImplementedError(f"{name} not supported")

    def _suffix(text, name):
        if 'qwen' in name:
            return add_chat_template_suffix_qwen(text)
        elif 'llama' in name:
            return add_chat_template_suffix_llama(text)
        elif 'gemma' in name:
            return add_chat_template_suffix_gemma(text)
        elif 'ministral' in name:
            return add_chat_template_suffix_ministral(text)
        elif 'olmo' in name:
            return add_chat_template_suffix_olmo(text)
        raise NotImplementedError(f"{name} not supported")

    def chat_prefix(text, add_system=False):
        return _prefix(text, add_system, model_name_lower)

    def chat_suffix(text):
        return _suffix(text, model_name_lower)

    # ── build_agent_context ────────────────────────────────────────────────────
    def build_agent_context(
        other_dialogues: List[Dict],
        tokenizer,
        bot_token: str,
        eot_token: str,
        latent_token: str,
        num_latent: int,
    ):
        """
        Build encoder context and teacher text context from other agents' previous-round data.

        other_dialogues: list of dicts, each with keys 'cot' and 'answer', one per other agent.

        Returns:
            context_tokens:      token ids with (N-1)*num_latent latent placeholders (student encoder input)
            text_context_tokens: plain text version of the context (teacher full_ref prefix)
        """
        OTHER_TEMPLATE = [
            "These are the solutions to the problem from other agents: {content}\n\n"
            "Using the responses from other agents as additional information, can you give an answer "
            "to the original question?\nExamine your solution and that other agents step by step.\n"
            "Make sure to state your answer at the end of the response.",
            "These are the responses from other agents: {content}\n\n"
            "What is your answer to the original question? "
            "Make sure to state your answer at the end of the response.",
        ]
        template = random.choice(OTHER_TEMPLATE)
        other_latent = latent_token * num_latent

        latent_parts = ""
        text_parts = ""
        for d in other_dialogues:
            latent_parts += bot_token + other_latent + eot_token + d["answer"] + " "
            text_parts += f"{d['cot']} {d['answer']} "

        latent_response = template.format(content=latent_parts.strip())
        context = chat_prefix(latent_response, add_system=False)

        text_response = template.format(content=text_parts.strip())
        text_context = chat_prefix(text_response, add_system=False)

        return (
            tokenizer.encode(context, add_special_tokens=False),
            tokenizer.encode(text_context, add_special_tokens=False),
        )

    # ── preprocess_mas ─────────────────────────────────────────────────────────
    def preprocess_mas(
        dialogue_list: Sequence[list],
        tokenizer: transformers.PreTrainedTokenizer,
        bot_id: int,
        eot_id: int,
        latent_id: int,
        bot_token: str,
        eot_token: str,
        latent_token: str,
        num_latent: int,
        num_agents: int,
        mask_inter_res: bool,
    ) -> Dict:
        """
        Build per-agent tokenised data for all dialogues.

        dialogue_list: list of dialogue turn lists.
          Each turn list: [turn0_meta, turn1_agent0_r0, turn2_agent1_r0, ..., turnN*R_agent{N-1}_r{R-1}]
          turn0_meta has keys: question_list (list of N questions), mask_inter_res
          all other turns have keys: cot, answer, mask_inter_res
        """
        print("Tokenizing inputs... This may take some time...")

        answer_prompts = [
            torch.tensor(tokenizer.encode("The answer is:")),
            torch.tensor(tokenizer.encode("I would say:")),
        ]
        if answer_prompts[0][0] == tokenizer.bos_token_id:
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]

        # Per-agent output lists (one entry per dialogue)
        out = {
            i: {
                "encoder_ids":    [],
                "decoder_ids":    [],
                "labels":         [],
                "model_pos":      [],
                "ref_pos":        [],
                "full_ref_ids":   [],
                "full_ref_labels":[],
                "long_dec_ids":   [],
                "long_labels":    [],
            }
            for i in range(num_agents)
        }
        mask_inter_res_list = []

        for dialogue_turns in dialogue_list:
            sample_mask = dialogue_turns[0].get("mask_inter_res", mask_inter_res)
            initial_questions = dialogue_turns[0].get("question_list", [""] * num_agents)
            dialogue_turns = dialogue_turns[1:]  # drop metadata turn

            # Per-agent accumulators for this dialogue
            acc = {
                i: {
                    "encoder_ids": [], "decoder_ids": [], "labels": [],
                    "model_pos": [], "ref_pos": [],
                    "long_dec_ids": [], "long_labels": [],
                    "full_ref_tokens": [], "response_spans": [],
                    "cum_len": 0,
                }
                for i in range(num_agents)
            }

            for turn_idx, turn_data in enumerate(dialogue_turns):
                agent_i = turn_idx % num_agents
                round_r = turn_idx // num_agents
                a = acc[agent_i]

                cot = turn_data["cot"]
                answer = turn_data["answer"]

                if round_r == 0:
                    # First round: no context from other agents
                    question_text = chat_prefix(initial_questions[agent_i], add_system=True)
                    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
                    context_tokens = []
                    text_ctx_tokens = []
                else:
                    # Use all other agents' round (r-1) data as context
                    prev_round_start = (round_r - 1) * num_agents
                    other_dialogues = [
                        dialogue_turns[prev_round_start + j]
                        for j in range(num_agents) if j != agent_i
                    ]
                    context_tokens, text_ctx_tokens = build_agent_context(
                        other_dialogues, tokenizer,
                        bot_token, eot_token, latent_token, num_latent,
                    )
                    question_tokens = []
                    question_text = ""

                # ── Student encoder ──────────────────────────────────────────
                encoder_ids = torch.tensor(
                    context_tokens + question_tokens + [bot_id], dtype=torch.long
                )
                a["encoder_ids"].append(encoder_ids)

                # ── Student decoder ──────────────────────────────────────────
                answer_text = chat_suffix(answer)
                answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
                decoder_ids = torch.tensor([eot_id] + answer_tokens, dtype=torch.long)
                a["decoder_ids"].append(decoder_ids)
                a["labels"].append(decoder_ids.clone())
                a["model_pos"].append(
                    get_answer_token_position(decoder_ids, answer_prompts, tokenizer)
                )

                # ── Teacher full_ref ─────────────────────────────────────────
                # Sequence: [other agents' round r-1 plain text] + [this agent's cot + answer]
                if round_r == 0:
                    ref_text = question_text + cot + answer_text
                else:
                    ref_text = cot + answer_text
                ref_tokens = tokenizer.encode(ref_text, add_special_tokens=False)
                ref_tokens = text_ctx_tokens + ref_tokens

                ref_tensor = torch.tensor(ref_tokens, dtype=torch.long)
                local_ref_pos = get_answer_token_position(ref_tensor, answer_prompts, tokenizer)
                a["ref_pos"].append(a["cum_len"] + local_ref_pos)

                ans_start = a["cum_len"] + len(ref_tokens) - len(answer_tokens)
                ans_end = a["cum_len"] + len(ref_tokens)
                a["response_spans"].append((ans_start, ans_end))
                a["cum_len"] += len(ref_tokens)
                a["full_ref_tokens"].extend(ref_tokens)

                # ── External decoder (latent → CoT) ─────────────────────────
                cot_tokens = tokenizer.encode(cot, add_special_tokens=False)
                long_dec = torch.tensor(cot_tokens, dtype=torch.long)
                long_lab = torch.cat([
                    torch.tensor([IGNORE_INDEX] * (num_latent + 1)),
                    long_dec,
                ], dim=0)
                a["long_dec_ids"].append(long_dec)
                a["long_labels"].append(long_lab)

            # ── Finalise full_ref tensors for each agent ─────────────────────
            for i in range(num_agents):
                a = acc[i]
                full_ref_ids = torch.tensor(a["full_ref_tokens"], dtype=torch.long)
                full_ref_labels = torch.full_like(full_ref_ids, IGNORE_INDEX)
                spans = a["response_spans"][-1:] if sample_mask else a["response_spans"]
                for start, end in spans:
                    full_ref_labels[start:end] = full_ref_ids[start:end]

                out[i]["encoder_ids"].append(a["encoder_ids"])
                out[i]["decoder_ids"].append(a["decoder_ids"])
                out[i]["labels"].append(a["labels"])
                out[i]["model_pos"].append(a["model_pos"])
                out[i]["ref_pos"].append(a["ref_pos"])
                out[i]["full_ref_ids"].append(full_ref_ids)
                out[i]["full_ref_labels"].append(full_ref_labels)
                out[i]["long_dec_ids"].append(a["long_dec_ids"])
                out[i]["long_labels"].append(a["long_labels"])

            mask_inter_res_list.append(torch.tensor(sample_mask, dtype=torch.bool))

        result = {}
        for i in range(num_agents):
            p = f"agent{i}_"
            d = out[i]
            result[f"{p}encoder_input_ids_list"]      = d["encoder_ids"]
            result[f"{p}decoder_input_ids_list"]      = d["decoder_ids"]
            result[f"{p}labels_list"]                 = d["labels"]
            result[f"{p}model_answer_position"]       = d["model_pos"]
            result[f"{p}ref_answer_position"]         = d["ref_pos"]
            result[f"{p}full_ref_input_ids"]          = d["full_ref_ids"]
            result[f"{p}full_ref_labels"]             = d["full_ref_labels"]
            result[f"{p}long_decoder_input_ids_list"] = d["long_dec_ids"]
            result[f"{p}long_labels_list"]            = d["long_labels"]
        result["mask_inter_res"] = mask_inter_res_list
        return result

    # ── Dataset ────────────────────────────────────────────────────────────────
    class MASSupervisedDataset(Dataset):
        QUESTION_COM_PROMPT = (
            "\nAnswer the above question. Provide any details you want at first and then "
            "give your final answer, decision or conclusion."
        )

        def __init__(
            self,
            data_name, raw_data, tokenizer,
            bot_id, eot_id, latent_id,
            bot_token, eot_token, latent_token,
            num_latent, num_agents,
            fixed_num_turns=2,
            model_name=None,
            mask_inter_res=False,
        ):
            super().__init__()
            logging.warning("Formatting inputs...")
            self.data_name = data_name
            dialogue_list = []

            for num_iter, example in enumerate(raw_data):
                if training_args.exp_mode and num_iter > training_args.exp_data_num:
                    break

                # messages[0] = metadata, messages[1..N*R] = per-agent per-round responses
                total_msg = 1 + num_agents * fixed_num_turns
                messages = example["messages"][:total_msg]
                if len(messages) < total_msg:
                    continue

                turns = []
                for i in range(len(messages)):
                    if i == 0:
                        # Read per-agent questions (1-indexed in data: question_agent1...question_agentN)
                        question_list = [
                            f"{messages[i].get(f'question_agent{j + 1}', '')}"
                            for j in range(num_agents)
                        ]
                    else:
                        question_list = [""] * num_agents

                    short_answer = f"{messages[i]['short_answer']}"
                    long_answer = f"{messages[i]['long_answer']}"
                    cot_parts = long_answer.split(". ")
                    if not training_args.include_last_cot:
                        cot_parts = cot_parts[:-1]
                    cot = ". ".join(cot_parts) + ".\n" if cot_parts else ""

                    answer = extract_short_answer(short_answer)
                    answer = f"The answer is: {answer}".replace("####", "")

                    sample_mask = example.get("mask_inter_res", mask_inter_res)
                    turns.append({
                        "question_list": question_list,
                        "cot": cot,
                        "answer": answer,
                        "mask_inter_res": sample_mask,
                    })

                # Filter by token budget
                all_text = " ".join(
                    t["question_list"][0] + t["cot"] + t["answer"] for t in turns
                )
                if len(tokenizer.encode(all_text)) > training_args.max_token_num:
                    continue

                if turns:
                    dialogue_list.append(turns)

            if training_args.exp_mode:
                dialogue_list = dialogue_list[:training_args.exp_data_num]

            print(
                f"{len(dialogue_list)} dialogues with "
                f"{sum(len(d) for d in dialogue_list)} turns in total..."
            )
            logging.warning("Tokenizing inputs... This may take some time...")

            self.data_dict = preprocess_mas(
                dialogue_list, tokenizer,
                bot_id, eot_id, latent_id,
                bot_token, eot_token, latent_token,
                num_latent, num_agents,
                mask_inter_res,
            )
            self.keys = list(self.data_dict.keys())

        def __len__(self):
            return len(self.data_dict["agent0_full_ref_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    # ── DataCollator ───────────────────────────────────────────────────────────
    @dataclass
    class DataCollatorForSupervisedDataset:
        tokenizer: transformers.PreTrainedTokenizer
        num_agents: int = 2

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            batch_size = len(instances)
            max_turns = max(
                len(inst["agent0_encoder_input_ids_list"]) for inst in instances
            )

            result = {}

            for agent_i in range(self.num_agents):
                p = f"agent{agent_i}_"

                # ── Pad turn lists to max_turns ────────────────────────────────
                enc_batch, dec_batch, lab_batch = [], [], []
                mpos_batch, rpos_batch = [], []
                ldec_batch, llab_batch = [], []

                for inst in instances:
                    n = len(inst[f"{p}encoder_input_ids_list"])
                    pad_n = max_turns - n
                    enc_batch.append(
                        inst[f"{p}encoder_input_ids_list"] +
                        [torch.zeros(1, dtype=torch.long)] * pad_n
                    )
                    dec_batch.append(
                        inst[f"{p}decoder_input_ids_list"] +
                        [torch.zeros(1, dtype=torch.long)] * pad_n
                    )
                    lab_batch.append(
                        inst[f"{p}labels_list"] +
                        [torch.full((1,), IGNORE_INDEX, dtype=torch.long)] * pad_n
                    )
                    mpos_batch.append(inst[f"{p}model_answer_position"] + [0] * pad_n)
                    rpos_batch.append(inst[f"{p}ref_answer_position"] + [0] * pad_n)
                    ldec_batch.append(
                        inst[f"{p}long_decoder_input_ids_list"] +
                        [torch.zeros(1, dtype=torch.long)] * pad_n
                    )
                    llab_batch.append(
                        inst[f"{p}long_labels_list"] +
                        [torch.full((1,), IGNORE_INDEX, dtype=torch.long)] * pad_n
                    )

                # ── Compute max sequence lengths across all turns and samples ──
                max_enc_len = max_dec_len = max_ldec_len = max_llab_len = 0
                for t in range(max_turns):
                    for b in range(batch_size):
                        max_enc_len  = max(max_enc_len,  enc_batch[b][t].size(0))
                        max_dec_len  = max(max_dec_len,  dec_batch[b][t].size(0))
                        max_ldec_len = max(max_ldec_len, ldec_batch[b][t].size(0))
                        max_llab_len = max(max_llab_len, llab_batch[b][t].size(0))

                # ── Pad each turn and collect ──────────────────────────────────
                enc_final = []
                dec_final = []
                lab_final = []
                ldec_final = []
                llab_final = []

                pad_id = self.tokenizer.pad_token_id

                for t in range(max_turns):
                    turn_enc  = [enc_batch[b][t]  for b in range(batch_size)]
                    turn_dec  = [dec_batch[b][t]  for b in range(batch_size)]
                    turn_lab  = [lab_batch[b][t]  for b in range(batch_size)]
                    turn_ldec = [ldec_batch[b][t] for b in range(batch_size)]
                    turn_llab = [llab_batch[b][t] for b in range(batch_size)]

                    # Encoder: left-pad (so that the [bot] token is always at the right end)
                    rev = [s.flip(0) for s in turn_enc]
                    pe = torch.nn.utils.rnn.pad_sequence(
                        rev, batch_first=True, padding_value=pad_id
                    ).flip(1)
                    if pe.size(1) < max_enc_len:
                        pe = torch.cat([
                            torch.full((batch_size, max_enc_len - pe.size(1)), pad_id, dtype=torch.long),
                            pe,
                        ], dim=1)

                    # Decoder / labels: right-pad
                    pd = torch.nn.utils.rnn.pad_sequence(
                        turn_dec, batch_first=True, padding_value=pad_id
                    )
                    if pd.size(1) < max_dec_len:
                        pd = torch.cat([
                            pd,
                            torch.full((batch_size, max_dec_len - pd.size(1)), pad_id, dtype=torch.long),
                        ], dim=1)

                    pl = torch.nn.utils.rnn.pad_sequence(
                        turn_lab, batch_first=True, padding_value=IGNORE_INDEX
                    )
                    if pl.size(1) < max_dec_len:
                        pl = torch.cat([
                            pl,
                            torch.full((batch_size, max_dec_len - pl.size(1)), IGNORE_INDEX, dtype=torch.long),
                        ], dim=1)

                    pld = torch.nn.utils.rnn.pad_sequence(
                        turn_ldec, batch_first=True, padding_value=pad_id
                    )
                    if pld.size(1) < max_ldec_len:
                        pld = torch.cat([
                            pld,
                            torch.full((batch_size, max_ldec_len - pld.size(1)), pad_id, dtype=torch.long),
                        ], dim=1)

                    pll = torch.nn.utils.rnn.pad_sequence(
                        turn_llab, batch_first=True, padding_value=IGNORE_INDEX
                    )
                    if pll.size(1) < max_llab_len:
                        pll = torch.cat([
                            pll,
                            torch.full((batch_size, max_llab_len - pll.size(1)), IGNORE_INDEX, dtype=torch.long),
                        ], dim=1)

                    enc_final.append(pe)
                    dec_final.append(pd)
                    lab_final.append(pl)
                    ldec_final.append(pld)
                    llab_final.append(pll)

                # Stack [T, B, L] → transpose → [B, T, L]
                enc_stacked  = torch.stack(enc_final).transpose(0, 1)
                dec_stacked  = torch.stack(dec_final).transpose(0, 1)
                lab_stacked  = torch.stack(lab_final).transpose(0, 1)
                ldec_stacked = torch.stack(ldec_final).transpose(0, 1)
                llab_stacked = torch.stack(llab_final).transpose(0, 1)

                # full_ref: pad across batch (variable-length per dialogue)
                full_ref_ids    = [inst[f"{p}full_ref_input_ids"]  for inst in instances]
                full_ref_labels = [inst[f"{p}full_ref_labels"]     for inst in instances]
                full_ref_ids_padded = torch.nn.utils.rnn.pad_sequence(
                    full_ref_ids, batch_first=True, padding_value=pad_id
                )
                full_ref_labels_padded = torch.nn.utils.rnn.pad_sequence(
                    full_ref_labels, batch_first=True, padding_value=IGNORE_INDEX
                )

                model_pos_tensor = torch.tensor(mpos_batch, dtype=torch.long)
                ref_pos_tensor   = torch.tensor(rpos_batch, dtype=torch.long)

                result[f"{p}encoder_input_ids_list"]      = enc_stacked
                result[f"{p}decoder_input_ids_list"]      = dec_stacked
                result[f"{p}labels_list"]                 = lab_stacked
                result[f"{p}encoder_attention_mask_list"] = enc_stacked.ne(pad_id)
                result[f"{p}model_answer_position"]       = model_pos_tensor
                result[f"{p}ref_answer_position"]         = ref_pos_tensor
                result[f"{p}full_ref_input_ids"]          = full_ref_ids_padded
                result[f"{p}full_ref_labels"]             = full_ref_labels_padded
                result[f"{p}full_ref_attention_mask"]     = full_ref_ids_padded.ne(pad_id)
                result[f"{p}long_decoder_input_ids_list"] = ldec_stacked
                result[f"{p}long_labels_list"]            = llab_stacked

            result["mask_inter_res"] = torch.stack(
                [inst["mask_inter_res"] for inst in instances]
            )
            return result

    # ── Data module ────────────────────────────────────────────────────────────
    def make_supervised_data_module(tokenizer, data_args, model_args, training_args) -> Dict:
        logging.warning("Downloading Data")
        dataset = load_dataset("json", data_files=data_args.data_name)["train"]
        train_dataset = MASSupervisedDataset(
            data_name=data_args.data_name,
            raw_data=dataset,
            tokenizer=tokenizer,
            bot_id=model.bot_id,
            eot_id=model.eot_id,
            latent_id=model.latent_id,
            bot_token=model.bot_token,
            eot_token=model.eot_token,
            latent_token=model.latent_token,
            num_latent=model.num_latent,
            num_agents=data_args.num_agents,
            fixed_num_turns=data_args.fixed_num_turns,
            model_name=model_args.model_name_or_path.split('/')[-1].lower(),
            mask_inter_res=training_args.mask_inter_res,
        )
        data_collator = DataCollatorForSupervisedDataset(
            tokenizer=tokenizer,
            num_agents=data_args.num_agents,
        )
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    # ── Launch training ────────────────────────────────────────────────────────
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
    )
    training_args.learning_rate = float(training_args.learning_rate)
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        type=str
    )
    args = parser.parse_args()
    train(args)
