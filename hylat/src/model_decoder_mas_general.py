"""
Generic N-agent HyLaT for multi-agent multi-turn dialogs.
Generalizes model_decoder_mas_dual.py from 2 agents to N agents.

Key differences from the dual version:
- forward() accepts agent{i}_* kwargs for any N >= 2 (0-indexed)
- Each agent in round r injects concatenated latents from all N-1 other agents' round r-1
- Teacher task uses correct round r-1 context (symmetric with student latent injection)
"""
import re
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from peft import get_peft_model
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    full_precision: bool = field(default=True, metadata={"help": "whether use int4 for the base model"})
    train: bool = field(default=True)
    lora_init: bool = field(default=False)
    token: Optional[str] = field(default=None)
    adapter_name_or_path: Optional[str] = field(default=None)
    lora_alpha: int = field(default=16)
    ckpt_dir: Optional[str] = field(default=None)
    stage1_ckpt_dir: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_name: str = field(default=None)
    debug_data: bool = field(default=False)
    batch_size: int = field(default=1)
    fixed_num_turns: int = field(default=2, metadata={"help": "number of rounds per agent"})
    num_agents: int = field(default=2, metadata={"help": "number of agents in MAS"})
    output_path: str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=28000)
    restore_from: str = field(default="")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    expt_name: str = field(default="default")
    icot_train_path: str = field(default="")
    num_latent: int = field(default=5)
    use_lora: bool = field(default=True)
    greedy: bool = field(default=False)
    exp_mode: bool = field(default=False)
    exp_data_num: int = field(default=10000)
    use_prj: bool = field(default=False)
    prj_dim: int = field(default=2048)
    prj_dropout: float = field(default=0.0)
    prj_no_ln: bool = field(default=False)
    distill_loss_div_std: bool = field(default=False)
    distill_loss_type: str = field(default="smooth_l1")
    distill_loss_factor: float = field(default=1.0)
    ref_loss_factor: float = field(default=1.0)
    inf_latent_iterations: int = field(default=1)
    inf_num_iterations: int = field(default=5)
    remove_eos: bool = field(default=False)
    print_ref_model_stats: bool = field(default=False)
    include_last_cot: bool = field(default=False)
    fix_attn_mask: bool = field(default=False)
    log_full: bool = field(default=False)
    print_loss: bool = field(default=True)
    max_token_num: int = field(default=1000)
    decode_loss_factor: float = field(default=0.1)
    use_decode: bool = field(default=False)
    mask_inter_res: bool = field(default=False)


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total:.2f}")


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


class HyLAT(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        dtype = torch.float16 if not training_args.bf16 else torch.bfloat16
        quant_cfg = None if model_args.full_precision else transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4',
        )
        self.codi = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            resume_download=True,
            **({"quantization_config": quant_cfg} if quant_cfg else {}),
        )

        ori_vocab_size = self.codi.config.vocab_size
        self.training = model_args.train

        # Special token IDs appended after the original vocabulary
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.eot_id = ori_vocab_size + 2
        self.ae_id = ori_vocab_size + 3
        self.latent_id = ori_vocab_size + 4

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        dummy_tokens = [f"<dummy_{i}>" for i in range(ori_vocab_size - len(self.tokenizer))]
        if dummy_tokens:
            self.tokenizer.add_tokens(dummy_tokens, special_tokens=False)
        self.tokenizer.add_special_tokens({
            "pad_token": "<pad>",
            "additional_special_tokens": ["<bot>", "<eot>", "<ae>", "<latent>"],
        })
        self.bot_token = "<bot>"
        self.eot_token = "<eot>"
        self.latent_token = "<latent>"
        self.codi.resize_token_embeddings(ori_vocab_size + 5)

        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent

        if training_args.use_lora:
            self.codi = get_peft_model(self.codi, lora_config)

        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))
            self.prj.to(self.codi.dtype)

        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError(f"Unknown distill_loss_type: {self.distill_loss_type}")

        self.use_decode = training_args.use_decode
        self.decode_loss_factor = training_args.decode_loss_factor
        self.fix_attn_mask = training_args.fix_attn_mask
        self.mask_inter_res = training_args.mask_inter_res

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.pad_token_id

        if self.training:
            self.init()

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name:
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name:
                try:
                    return model.get_base_model().transformer.wte
                except Exception:
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception:
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError

    def init(self):
        print_trainable_parameters(self)
        if self.training_args.restore_from:
            print(f"Loading from checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print("Done.")

    def decoder_forward(self, latent_seq, long_decoder_input_ids, long_labels):
        batch_size = latent_seq.size(0)
        long_answer_embeds = self.get_embd(self.decoder, self.model_name)(long_decoder_input_ids)
        ae_embeds = self.get_embd(self.decoder, self.model_name)(
            torch.tensor([self.ae_id]).to(long_decoder_input_ids)
        ).unsqueeze(0)
        inputs_embeds = torch.cat([latent_seq, ae_embeds.expand(batch_size, -1, -1), long_answer_embeds], dim=1)
        outputs = self.decoder(inputs_embeds=inputs_embeds, output_hidden_states=False, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        return self.loss_fct(logits.reshape(-1, logits.size(-1)), long_labels[:, 1:].reshape(-1))

    def forward(
        self,
        step: int = None,
        step_ratio: float = None,
        mask_inter_res: torch.BoolTensor = None,
        **kwargs,
    ):
        # Infer num_agents from batch keys (agent0_full_ref_input_ids, agent1_..., ...)
        num_agents = sum(1 for k in kwargs if re.match(r'^agent\d+_full_ref_input_ids$', k))
        assert num_agents >= 2, f"Need at least 2 agents, got {num_agents}"

        # ── Collect per-agent tensors ──────────────────────────────────────────
        def _get(i, suffix):
            return kwargs[f"agent{i}_{suffix}"]

        agents_encoder    = [_get(i, "encoder_input_ids_list")       for i in range(num_agents)]
        agents_decoder    = [_get(i, "decoder_input_ids_list")       for i in range(num_agents)]
        agents_labels     = [_get(i, "labels_list")                  for i in range(num_agents)]
        agents_enc_mask   = [_get(i, "encoder_attention_mask_list")  for i in range(num_agents)]
        agents_model_pos  = [_get(i, "model_answer_position")        for i in range(num_agents)]
        agents_ref_pos    = [_get(i, "ref_answer_position")          for i in range(num_agents)]
        agents_full_ref   = [_get(i, "full_ref_input_ids")           for i in range(num_agents)]
        agents_full_lab   = [_get(i, "full_ref_labels")              for i in range(num_agents)]
        agents_full_mask  = [_get(i, "full_ref_attention_mask")      for i in range(num_agents)]
        agents_long_dec   = [_get(i, "long_decoder_input_ids_list")  for i in range(num_agents)]
        agents_long_lab   = [_get(i, "long_labels_list")             for i in range(num_agents)]

        device = agents_full_ref[0].device
        batch_size = agents_encoder[0].size(0)
        num_turns = agents_encoder[0].size(1)

        loss_detail = {
            f"agent{i}_{m}": []
            for i in range(num_agents)
            for m in ["teacher_ce", "student_ce", "distill", "decode", "total"]
        }

        # Transpose [B, T, L] -> [T, B, L] for per-turn indexing
        for i in range(num_agents):
            agents_encoder[i]  = agents_encoder[i].transpose(0, 1)
            agents_decoder[i]  = agents_decoder[i].transpose(0, 1)
            agents_labels[i]   = agents_labels[i].transpose(0, 1)
            agents_enc_mask[i] = agents_enc_mask[i].transpose(0, 1)
            agents_long_dec[i] = agents_long_dec[i].transpose(0, 1)
            agents_long_lab[i] = agents_long_lab[i].transpose(0, 1)

        # ── Teacher Task ───────────────────────────────────────────────────────
        # Each agent has a full text sequence; train with teacher-forcing.
        # Loss is masked to only the agent's own answer spans.
        agents_ref_nograd = []
        agents_ref_grad   = []
        total_ref_loss = torch.tensor(0.0, device=device)

        for i in range(num_agents):
            attn_mask = agents_full_mask[i] if self.fix_attn_mask else None
            with torch.no_grad():
                ref_out = self.codi(
                    input_ids=agents_full_ref[i],
                    output_hidden_states=True,
                    attention_mask=attn_mask,
                )
            ref_out_grad = self.codi(
                input_ids=agents_full_ref[i],
                output_hidden_states=True,
                attention_mask=attn_mask,
            )
            agents_ref_nograd.append(ref_out)
            agents_ref_grad.append(ref_out_grad)

            ref_logits = ref_out_grad.logits
            ref_loss = self.loss_fct(
                ref_logits[:, :-1, :].reshape(-1, ref_logits.size(-1)),
                agents_full_lab[i][:, 1:].reshape(-1),
            ) * self.ref_loss_factor
            total_ref_loss = total_ref_loss + ref_loss

            # Per-turn teacher CE (logging only)
            shift_logits = ref_logits[:, :-1, :]
            shift_labels = agents_full_lab[i][:, 1:]
            token_ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).view(shift_labels.size())

            for t in range(num_turns):
                start = agents_ref_pos[i][:, t]
                end = agents_ref_pos[i][:, t + 1] - 5 if t < num_turns - 1 \
                    else [token_ce.size(1)] * batch_size
                turn_losses = [token_ce[b, start[b]:end[b]].mean() for b in range(batch_size)]
                loss_detail[f"agent{i}_teacher_ce"].append(torch.stack(turn_losses).mean().item())

        # ── Student Task ───────────────────────────────────────────────────────
        agents_past_kv    = [None] * num_agents
        agents_gen_latents = [[] for _ in range(num_agents)]  # [agent][turn] -> [B, num_latent, H]

        total_ce_loss      = torch.tensor(0.0, device=device)
        total_distill_loss = torch.tensor(0.0, device=device)
        total_decode_loss  = torch.tensor(0.0, device=device)

        for turn_idx in range(num_turns):
            is_last_turn = (turn_idx == num_turns - 1)
            if mask_inter_res is not None:
                should_compute = (~mask_inter_res) | is_last_turn  # [B]
            else:
                should_compute = torch.ones(batch_size, dtype=torch.bool, device=device)

            for agent_i in range(num_agents):
                enc_ids   = agents_encoder[agent_i][turn_idx]
                dec_ids   = agents_decoder[agent_i][turn_idx]
                labels    = agents_labels[agent_i][turn_idx]
                enc_mask  = agents_enc_mask[agent_i][turn_idx]
                long_dec  = agents_long_dec[agent_i][turn_idx]
                long_lab  = agents_long_lab[agent_i][turn_idx]

                model_pos = agents_model_pos[agent_i][:, turn_idx] - 1
                ref_pos   = agents_ref_pos[agent_i][:, turn_idx] - 1

                # 1. Build encoder embeddings; inject other agents' previous-turn latents
                enc_embeds = self.get_embd(self.codi, self.model_name)(enc_ids)
                if turn_idx > 0:
                    # Concatenate latents from all other agents' previous turn: [B, (N-1)*L, H]
                    other_latents = torch.cat(
                        [agents_gen_latents[j][turn_idx - 1]
                         for j in range(num_agents) if j != agent_i],
                        dim=1,
                    )
                    enc_embeds = self._inject_other_agent_latent(enc_ids, enc_embeds, other_latents)

                # 2. Extend attention mask to cover KV-cache from previous turns
                past_kv = agents_past_kv[agent_i]
                if past_kv is not None:
                    past_len = past_kv[0][0].size(2)
                    past_mask = torch.ones(
                        enc_embeds.size(0), past_len,
                        dtype=enc_mask.dtype, device=enc_mask.device,
                    )
                    enc_mask = torch.cat([past_mask, enc_mask], dim=1)

                # 3. Encode: question (+ injected context) → hidden state seed for latent
                out = self.codi(
                    inputs_embeds=enc_embeds,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_kv,
                    attention_mask=enc_mask,
                )
                agents_past_kv[agent_i] = out.past_key_values
                latent_embd = out.hidden_states[-1][:, -1, :].unsqueeze(1)
                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

                # 4. Autoregressively generate num_latent latent vectors
                latent_list = []
                for _ in range(self.num_latent):
                    out = self.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=agents_past_kv[agent_i],
                    )
                    agents_past_kv[agent_i] = out.past_key_values
                    latent_embd = out.hidden_states[-1][:, -1, :].unsqueeze(1)
                    if self.use_prj:
                        latent_embd = self.prj(latent_embd)
                    latent_list.append(latent_embd)

                current_latents = torch.cat(latent_list, dim=1)  # [B, num_latent, H]
                agents_gen_latents[agent_i].append(current_latents)

                # 5. Decode answer
                dec_embeds = self.get_embd(self.codi, self.model_name)(dec_ids)
                dyn_mask = None
                if self.fix_attn_mask:
                    dyn_mask = torch.cat([
                        enc_mask,
                        torch.ones((batch_size, self.num_latent), device=device),
                        torch.ones((batch_size, dec_embeds.size(1)), device=device),
                    ], dim=1).bool()

                out = self.codi(
                    inputs_embeds=dec_embeds,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=agents_past_kv[agent_i],
                    attention_mask=dyn_mask,
                )
                agents_past_kv[agent_i] = out.past_key_values

                # 6. CE loss (only for samples that should compute)
                logits = out.logits
                masked_labels = labels.clone()
                masked_labels[~should_compute] = IGNORE_INDEX
                if (masked_labels[:, 1:] != IGNORE_INDEX).sum() > 0:
                    ce_loss = self.loss_fct(
                        logits[:, :-1, :].reshape(-1, logits.size(-1)),
                        masked_labels[:, 1:].reshape(-1),
                    )
                else:
                    ce_loss = torch.tensor(0.0, device=device).to(logits)
                total_ce_loss = total_ce_loss + ce_loss

                # 7. Distillation loss: align student hidden state with teacher at answer position
                ref_out = agents_ref_nograd[agent_i]
                distill_loss = torch.tensor(0.0, device=device).to(ce_loss)
                for _, (out_layer, ref_layer) in enumerate(
                    zip(out.hidden_states, ref_out.hidden_states)
                ):
                    idx = ref_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_layer.size(-1))
                    ref_sel = ref_layer.gather(1, idx)
                    out_sel = out_layer.gather(
                        1,
                        model_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out_layer.size(-1)),
                    )
                    if should_compute.any():
                        d_tmp = self.distill_loss_fct(
                            out_sel[should_compute], ref_sel[should_compute].detach()
                        )
                        if self.distill_loss_div_std:
                            std = ref_sel[should_compute].std()
                            if std > 0:
                                if self.distill_loss_type == 'l2':
                                    d_tmp = d_tmp / std
                                d_tmp = d_tmp / std
                        distill_loss = distill_loss + d_tmp

                distill_loss = distill_loss / len(out.hidden_states) * self.distill_loss_factor
                total_distill_loss = total_distill_loss + distill_loss

                # 8. Decode loss: reconstruct CoT text from latent
                if self.use_decode:
                    decode_loss = self.decoder_forward(current_latents, long_dec, long_lab)
                else:
                    decode_loss = torch.tensor(0).to(ce_loss)
                decode_loss = decode_loss * self.decode_loss_factor
                total_decode_loss = total_decode_loss + decode_loss

                turn_total = ce_loss + distill_loss + decode_loss
                loss_detail[f"agent{agent_i}_student_ce"].append(ce_loss.detach().item())
                loss_detail[f"agent{agent_i}_distill"].append(distill_loss.detach().item())
                loss_detail[f"agent{agent_i}_decode"].append(decode_loss.detach().item())
                loss_detail[f"agent{agent_i}_total"].append(turn_total.detach().item())

        loss = total_ce_loss + total_distill_loss + total_ref_loss + total_decode_loss

        if self.print_loss:
            print(
                f"total_ce={total_ce_loss:.4f}, distill={total_distill_loss:.4f}, "
                f"ref={total_ref_loss:.4f}, decode={total_decode_loss:.4f}"
            )

        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": total_ce_loss.detach().item(),
            "distill_loss": total_distill_loss.detach().item(),
            "ref_ce_loss": total_ref_loss.detach().item(),
            "decode_loss": total_decode_loss.detach().item(),
            "loss_detail": loss_detail,
        }

    def _inject_other_agent_latent(
        self,
        encoder_input_ids: torch.Tensor,  # [B, L]
        encoder_embeds: torch.Tensor,     # [B, L, H]
        other_latents: torch.Tensor,      # [B, (N-1)*num_latent, H]
    ):
        """Fill all <latent> positions in encoder_embeds with other agents' latents."""
        for i in range(other_latents.size(0)):
            latent_flag = encoder_input_ids[i] == self.latent_id
            encoder_embeds[i][latent_flag] = other_latents[i].to(encoder_embeds).detach()
        return encoder_embeds
