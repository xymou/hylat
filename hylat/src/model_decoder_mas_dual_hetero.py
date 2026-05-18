"""
Heterogeneous multi-agent communication.

Two different backbone models (e.g. Llama + Qwen) communicate via learned bidirectional
adapters that project latents between the two hidden spaces.

Architecture:
  Agent1 (codi1) ──► latent [B, num_latent, H1]
                            │
                       adapter_1to2 (H1→H2)
                            │
  Agent2 (codi2) ◄── projected latent [B, num_latent, H2]

  and symmetrically adapter_2to1 (H2→H1) for the reverse direction.

Adapters receive gradients via the downstream CE loss of the receiving agent.
Source-model gradients are detached at the injection boundary.
"""
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from peft import get_peft_model
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    model2_name_or_path: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "LoRA rank for Agent1"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha for Agent1"})
    lora_r2: int = field(default=128, metadata={"help": "LoRA rank for Agent2"})
    lora_alpha2: int = field(default=16, metadata={"help": "LoRA alpha for Agent2"})
    lora_dropout: float = field(default=0.05)
    full_precision: bool = field(default=True)
    train: bool = field(default=True)
    lora_init: bool = field(default=False)
    token: Optional[str] = field(default=None)
    adapter_name_or_path: Optional[str] = field(default=None)
    ckpt_dir: Optional[str] = field(default=None)
    stage1_ckpt_dir: Optional[str] = field(default=None, metadata={"help": "Stage1 checkpoint for Agent1"})
    stage1_ckpt_dir2: Optional[str] = field(default=None, metadata={"help": "Stage1 checkpoint for Agent2"})
    freeze_model1: bool = field(default=False, metadata={"help": "Freeze Agent1 backbone; train only adapter"})
    freeze_model2: bool = field(default=False, metadata={"help": "Freeze Agent2 backbone; train only adapter"})


@dataclass
class DataArguments:
    data_name: str = field(default=None, metadata={"help": "Path to training data."})
    debug_data: bool = field(default=False)
    batch_size: int = field(default=1)
    fixed_num_turns: int = field(default=2)
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
    prj_dim2: int = field(default=0, metadata={"help": "prj hidden dim for Agent2. 0 = same as prj_dim."})
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
    adapter_hidden_dim: int = field(
        default=0,
        metadata={"help": "Hidden dim for the cross-model adapter MLP. 0 = linear + LayerNorm."}
    )
    adapter_lr: float = field(
        default=0.0,
        metadata={"help": "Learning rate for adapter_1to2 and adapter_2to1. 0 = same as learning_rate."}
    )


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable} || total: {total} || trainable%: {100 * trainable / total:.2f}")


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def _make_adapter(in_dim: int, out_dim: int, hidden_dim: int, dtype) -> nn.Module:
    """Build a bidirectional adapter. hidden_dim=0 → linear + LayerNorm."""
    if hidden_dim > 0:
        adapter = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
    else:
        adapter = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
    return adapter.to(dtype)


class HyLAT(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config1=None, lora_config2=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model1_name = model_args.model_name_or_path
        self.model2_name = model_args.model2_name_or_path

        dtype = torch.bfloat16 if training_args.bf16 else torch.float16

        # ── Load model1 (Agent1) ──────────────────────────────────────────────
        load_kwargs = dict(torch_dtype=dtype, resume_download=True)
        if not model_args.full_precision:
            load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            )
        self.codi1 = AutoModelForCausalLM.from_pretrained(self.model1_name, **load_kwargs)
        self.tokenizer1 = AutoTokenizer.from_pretrained(self.model1_name, use_fast=False)

        ori_vocab1 = self.codi1.config.vocab_size
        dummy_tokens1 = [f"<dummy_{i}>" for i in range(ori_vocab1 - len(self.tokenizer1))]
        if dummy_tokens1:
            self.tokenizer1.add_tokens(dummy_tokens1, special_tokens=False)
        self.tokenizer1.add_special_tokens({
            "pad_token": "<pad>",
            "additional_special_tokens": ["<bot>", "<eot>", "<ae>", "<latent>"],
        })
        self.pad_token_id1 = ori_vocab1
        self.bot_id1      = ori_vocab1 + 1
        self.eot_id1      = ori_vocab1 + 2
        self.ae_id1       = ori_vocab1 + 3
        self.latent_id1   = ori_vocab1 + 4
        self.bot_token1   = "<bot>"
        self.eot_token1   = "<eot>"
        self.latent_token = "<latent>"
        self.codi1.resize_token_embeddings(ori_vocab1 + 5)

        # ── Load model2 (Agent2) ──────────────────────────────────────────────
        self.codi2 = AutoModelForCausalLM.from_pretrained(self.model2_name, **load_kwargs)
        self.tokenizer2 = AutoTokenizer.from_pretrained(self.model2_name, use_fast=False)

        ori_vocab2 = self.codi2.config.vocab_size
        dummy_tokens2 = [f"<dummy_{i}>" for i in range(ori_vocab2 - len(self.tokenizer2))]
        if dummy_tokens2:
            self.tokenizer2.add_tokens(dummy_tokens2, special_tokens=False)
        self.tokenizer2.add_special_tokens({
            "pad_token": "<pad>",
            "additional_special_tokens": ["<bot>", "<eot>", "<ae>", "<latent>"],
        })
        self.pad_token_id2 = ori_vocab2
        self.bot_id2       = ori_vocab2 + 1
        self.eot_id2       = ori_vocab2 + 2
        self.ae_id2        = ori_vocab2 + 3
        self.latent_id2    = ori_vocab2 + 4
        self.bot_token2    = "<bot>"
        self.eot_token2    = "<eot>"
        self.codi2.resize_token_embeddings(ori_vocab2 + 5)

        self.dim1 = self.codi1.config.hidden_size
        self.dim2 = self.codi2.config.hidden_size
        self.num_latent = training_args.num_latent

        # ── LoRA ──────────────────────────────────────────────────────────────
        if training_args.use_lora:
            if lora_config1 is not None:
                self.codi1 = get_peft_model(self.codi1, lora_config1)
            if lora_config2 is not None:
                self.codi2 = get_peft_model(self.codi2, lora_config2)

        # ── Optional freeze ───────────────────────────────────────────────────
        if model_args.freeze_model1:
            freeze_model(self.codi1)
        if model_args.freeze_model2:
            freeze_model(self.codi2)

        # ── Per-model projection layers (H→H, intra-model) ───────────────────
        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            prj_dim2 = training_args.prj_dim2 if training_args.prj_dim2 > 0 else training_args.prj_dim
            self.prj1 = self._make_prj(self.dim1, training_args.prj_dim, training_args, dtype)
            self.prj2 = self._make_prj(self.dim2, prj_dim2, training_args, dtype)

        # ── Bidirectional cross-model adapters ────────────────────────────────
        adapter_hidden = training_args.adapter_hidden_dim
        self.adapter_1to2 = _make_adapter(self.dim1, self.dim2, adapter_hidden, dtype)
        self.adapter_2to1 = _make_adapter(self.dim2, self.dim1, adapter_hidden, dtype)

        # ── Losses ────────────────────────────────────────────────────────────
        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        self.decode_loss_factor = training_args.decode_loss_factor
        self.use_decode = training_args.use_decode
        self.fix_attn_mask = training_args.fix_attn_mask
        self.mask_inter_res = training_args.mask_inter_res

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError(f"Unknown distill_loss_type: {self.distill_loss_type}")

        if self.model_args.train:
            self.init()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_prj(self, dim, prj_dim, training_args, dtype):
        layers = [
            nn.Dropout(training_args.prj_dropout),
            nn.Linear(dim, prj_dim),
            nn.GELU(),
            nn.Linear(prj_dim, dim),
        ]
        prj = nn.Sequential(*layers)
        if not training_args.prj_no_ln:
            prj.add_module("ln", nn.LayerNorm(dim))
        return prj.to(dtype)

    def get_embd(self, model, model_name):
        try:
            if "gpt2" in model_name:
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
            raise NotImplementedError(f"Cannot find embedding layer for {model_name}")

    def init(self):
        print_trainable_parameters(self)
        if self.training_args.restore_from:
            print(f"Loading from checkpoint: {self.training_args.restore_from}")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print("Loaded.")

    # ── Differentiable latent injection ──────────────────────────────────────

    def _inject_other_agent_latent(
        self,
        encoder_input_ids: torch.Tensor,   # [B, L]
        encoder_embeds: torch.Tensor,       # [B, L, H_target]
        other_latents: torch.Tensor,        # [B, num_latent, H_source]
        adapter: nn.Module,                 # H_source → H_target
        latent_id: int,
    ) -> torch.Tensor:
        """
        Replace the <latent> placeholder positions in encoder_embeds with
        other_latents projected through the adapter.

        Source-model gradients are detached; adapter weights receive gradients
        via the receiving agent's downstream CE loss.

        Latent tokens are always contiguous in the sequence (built as
        latent_token * num_latent in build_agent_context), so we use a
        concat-based approach that is fully differentiable without in-place ops.
        """
        batch_size = other_latents.size(0)
        projected = adapter(
            other_latents.detach().to(next(adapter.parameters()).dtype)
        ).to(encoder_embeds.dtype)  # [B, num_latent, H_target]

        parts = []
        for i in range(batch_size):
            positions = (encoder_input_ids[i] == latent_id).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                parts.append(encoder_embeds[i])
                continue
            n = len(positions)
            start = positions[0].item()
            end   = positions[-1].item() + 1
            emb_i = torch.cat([
                encoder_embeds[i, :start],       # [start, H]
                projected[i, :n],                # [n, H]  — has adapter gradient
                encoder_embeds[i, end:],         # [L-end, H]
            ], dim=0)
            parts.append(emb_i)
        return torch.stack(parts, dim=0)  # [B, L, H]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        # Agent1
        agent1_encoder_input_ids_list,
        agent1_decoder_input_ids_list,
        agent1_labels_list,
        agent1_encoder_attention_mask_list,
        agent1_model_answer_position,
        agent1_full_ref_input_ids,
        agent1_full_ref_labels,
        agent1_full_ref_attention_mask,
        agent1_ref_answer_position,
        agent1_long_decoder_input_ids_list,
        agent1_long_labels_list,
        # Agent2
        agent2_encoder_input_ids_list,
        agent2_decoder_input_ids_list,
        agent2_labels_list,
        agent2_encoder_attention_mask_list,
        agent2_model_answer_position,
        agent2_full_ref_input_ids,
        agent2_full_ref_labels,
        agent2_full_ref_attention_mask,
        agent2_ref_answer_position,
        agent2_long_decoder_input_ids_list,
        agent2_long_labels_list,
        step=None,
        step_ratio=None,
        mask_inter_res=None,
    ):
        dev = agent1_full_ref_input_ids.device
        batch_size = agent1_encoder_input_ids_list.size(0)
        num_turns  = agent1_encoder_input_ids_list.size(1)

        loss_detail = {
            "agent1_teacher_ce": [], "agent1_student_ce": [],
            "agent1_distill": [],    "agent1_decode": [],    "agent1_total": [],
            "agent2_teacher_ce": [], "agent2_student_ce": [],
            "agent2_distill": [],    "agent2_decode": [],    "agent2_total": [],
        }

        # Transpose [B, T, L] → [T, B, L]
        def tr(x): return x.transpose(0, 1)
        agent1_encoder_input_ids_list   = tr(agent1_encoder_input_ids_list)
        agent1_decoder_input_ids_list   = tr(agent1_decoder_input_ids_list)
        agent1_labels_list              = tr(agent1_labels_list)
        agent1_encoder_attention_mask_list = tr(agent1_encoder_attention_mask_list)
        agent1_long_decoder_input_ids_list = tr(agent1_long_decoder_input_ids_list)
        agent1_long_labels_list            = tr(agent1_long_labels_list)

        agent2_encoder_input_ids_list   = tr(agent2_encoder_input_ids_list)
        agent2_decoder_input_ids_list   = tr(agent2_decoder_input_ids_list)
        agent2_labels_list              = tr(agent2_labels_list)
        agent2_encoder_attention_mask_list = tr(agent2_encoder_attention_mask_list)
        agent2_long_decoder_input_ids_list = tr(agent2_long_decoder_input_ids_list)
        agent2_long_labels_list            = tr(agent2_long_labels_list)

        # ── Teacher task: each agent uses its own model ───────────────────────
        with torch.no_grad():
            agent1_ref_outputs_no_grad = self.codi1(
                input_ids=agent1_full_ref_input_ids,
                output_hidden_states=True,
                attention_mask=agent1_full_ref_attention_mask if self.fix_attn_mask else None,
            )
        agent1_ref_outputs = self.codi1(
            input_ids=agent1_full_ref_input_ids,
            output_hidden_states=True,
            attention_mask=agent1_full_ref_attention_mask if self.fix_attn_mask else None,
        )
        agent1_ref_logits = agent1_ref_outputs.logits
        agent1_ref_loss = self.loss_fct(
            agent1_ref_logits[:, :-1].reshape(-1, agent1_ref_logits.size(-1)),
            agent1_full_ref_labels[:, 1:].reshape(-1),
        ) * self.ref_loss_factor

        shift_logits1 = agent1_ref_logits[:, :-1]
        shift_labels1 = agent1_full_ref_labels[:, 1:]
        token_ce1 = F.cross_entropy(
            shift_logits1.reshape(-1, shift_logits1.size(-1)),
            shift_labels1.reshape(-1),
            reduction="none",
        ).view(shift_labels1.size())
        for t in range(num_turns):
            start = agent1_ref_answer_position[:, t]
            end = [token_ce1.size(1)] * batch_size if t == num_turns - 1 \
                  else agent1_ref_answer_position[:, t + 1] - 5
            turn_losses = [token_ce1[b, start[b]:end[b]].mean() for b in range(batch_size)]
            loss_detail["agent1_teacher_ce"].append(torch.stack(turn_losses).mean().item())

        with torch.no_grad():
            agent2_ref_outputs_no_grad = self.codi2(
                input_ids=agent2_full_ref_input_ids,
                output_hidden_states=True,
                attention_mask=agent2_full_ref_attention_mask if self.fix_attn_mask else None,
            )
        agent2_ref_outputs = self.codi2(
            input_ids=agent2_full_ref_input_ids,
            output_hidden_states=True,
            attention_mask=agent2_full_ref_attention_mask if self.fix_attn_mask else None,
        )
        agent2_ref_logits = agent2_ref_outputs.logits
        agent2_ref_loss = self.loss_fct(
            agent2_ref_logits[:, :-1].reshape(-1, agent2_ref_logits.size(-1)),
            agent2_full_ref_labels[:, 1:].reshape(-1),
        ) * self.ref_loss_factor

        shift_logits2 = agent2_ref_logits[:, :-1]
        shift_labels2 = agent2_full_ref_labels[:, 1:]
        token_ce2 = F.cross_entropy(
            shift_logits2.reshape(-1, shift_logits2.size(-1)),
            shift_labels2.reshape(-1),
            reduction="none",
        ).view(shift_labels2.size())
        for t in range(num_turns):
            start = agent2_ref_answer_position[:, t]
            end = [token_ce2.size(1)] * batch_size if t == num_turns - 1 \
                  else agent2_ref_answer_position[:, t + 1] - 1
            turn_losses = [token_ce2[b, start[b]:end[b]].mean() for b in range(batch_size)]
            loss_detail["agent2_teacher_ce"].append(torch.stack(turn_losses).mean().item())

        total_ref_loss = agent1_ref_loss + agent2_ref_loss

        # ── Student task ─────────────────────────────────────────────────────
        agent1_past_kv = None
        agent2_past_kv = None
        agent1_gen_latents: list = []
        agent2_gen_latents: list = []

        total_ce_loss     = 0
        total_distill_loss = 0
        total_decode_loss  = 0

        for turn_idx in range(num_turns):
            is_last_turn = (turn_idx == num_turns - 1)
            sample_should_compute = (
                (~mask_inter_res) | is_last_turn
                if mask_inter_res is not None
                else torch.ones(batch_size, dtype=torch.bool, device=dev)
            )

            # ── Agent1 ───────────────────────────────────────────────────────
            a1_enc_ids  = agent1_encoder_input_ids_list[turn_idx]
            a1_dec_ids  = agent1_decoder_input_ids_list[turn_idx]
            a1_labels   = agent1_labels_list[turn_idx]
            a1_enc_mask = agent1_encoder_attention_mask_list[turn_idx]
            a1_long_dec = agent1_long_decoder_input_ids_list[turn_idx]
            a1_long_lbl = agent1_long_labels_list[turn_idx]
            a1_model_pos = agent1_model_answer_position[:, turn_idx] - 1
            a1_ref_pos   = agent1_ref_answer_position[:, turn_idx] - 1

            a1_enc_embs = self.get_embd(self.codi1, self.model1_name)(a1_enc_ids)
            if turn_idx > 0 and agent2_gen_latents:
                a1_enc_embs = self._inject_other_agent_latent(
                    a1_enc_ids, a1_enc_embs,
                    agent2_gen_latents[turn_idx - 1],
                    self.adapter_2to1,
                    self.latent_id1,
                )

            if agent1_past_kv is not None:
                past_len = agent1_past_kv[0][0].size(2)
                past_mask = torch.ones(a1_enc_embs.size(0), past_len,
                                       dtype=a1_enc_mask.dtype, device=a1_enc_mask.device)
                a1_enc_mask = torch.cat([past_mask, a1_enc_mask], dim=1)

            out = self.codi1(inputs_embeds=a1_enc_embs, use_cache=True,
                             output_hidden_states=True, past_key_values=agent1_past_kv,
                             attention_mask=a1_enc_mask)
            agent1_past_kv = out.past_key_values
            latent_embd = out.hidden_states[-1][:, -1:, :]
            if self.use_prj:
                latent_embd = self.prj1(latent_embd)

            latent_list = []
            for _ in range(self.num_latent):
                out = self.codi1(inputs_embeds=latent_embd, use_cache=True,
                                 output_hidden_states=True, past_key_values=agent1_past_kv)
                agent1_past_kv = out.past_key_values
                latent_embd = out.hidden_states[-1][:, -1:, :]
                if self.use_prj:
                    latent_embd = self.prj1(latent_embd)
                latent_list.append(latent_embd)
            agent1_current_latents = torch.cat(latent_list, dim=1)  # [B, num_latent, H1]
            agent1_gen_latents.append(agent1_current_latents)

            a1_dec_embs = self.get_embd(self.codi1, self.model1_name)(a1_dec_ids)
            dyn_mask = None
            if self.fix_attn_mask:
                dyn_mask = torch.cat([
                    a1_enc_mask,
                    torch.ones(batch_size, self.num_latent, device=dev),
                    torch.ones(batch_size, a1_dec_embs.size(1), device=dev),
                ], dim=1).bool()

            out = self.codi1(inputs_embeds=a1_dec_embs, use_cache=True,
                             output_hidden_states=True, past_key_values=agent1_past_kv,
                             attention_mask=dyn_mask)
            agent1_past_kv = out.past_key_values

            # Agent1 CE loss
            logits = out.logits
            masked_labels = a1_labels.clone()
            masked_labels[~sample_should_compute] = IGNORE_INDEX
            valid = (masked_labels[:, 1:] != IGNORE_INDEX).sum()
            ce_loss = (
                self.loss_fct(logits[:, :-1].reshape(-1, logits.size(-1)),
                              masked_labels[:, 1:].reshape(-1))
                if valid > 0 else torch.tensor(0.0, device=dev).to(logits)
            )
            total_ce_loss = total_ce_loss + ce_loss

            # Agent1 distill loss (against its own teacher)
            distill_loss = torch.tensor(0.0, device=dev).to(ce_loss)
            for _, (student_hs, teacher_hs) in enumerate(
                    zip(out.hidden_states, agent1_ref_outputs_no_grad.hidden_states)):
                ref_sel = teacher_hs.gather(
                    1, a1_ref_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, teacher_hs.size(-1)))
                stu_sel = student_hs.gather(
                    1, a1_model_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, student_hs.size(-1)))
                if sample_should_compute.any():
                    dt = self.distill_loss_fct(stu_sel[sample_should_compute],
                                               ref_sel[sample_should_compute].detach())
                    if self.distill_loss_div_std:
                        std = ref_sel[sample_should_compute].std()
                        if std > 0:
                            dt = dt / std
                    distill_loss = distill_loss + dt
            distill_loss = distill_loss / len(out.hidden_states) * self.distill_loss_factor
            total_distill_loss = total_distill_loss + distill_loss

            decode_loss = torch.tensor(0, device=dev).to(ce_loss)
            decode_loss = decode_loss * self.decode_loss_factor
            total_decode_loss = total_decode_loss + decode_loss

            loss_detail["agent1_student_ce"].append(ce_loss.detach().item())
            loss_detail["agent1_distill"].append(distill_loss.detach().item())
            loss_detail["agent1_decode"].append(decode_loss.detach().item())
            loss_detail["agent1_total"].append((ce_loss + distill_loss + decode_loss).detach().item())

            # ── Agent2 ───────────────────────────────────────────────────────
            a2_enc_ids  = agent2_encoder_input_ids_list[turn_idx]
            a2_dec_ids  = agent2_decoder_input_ids_list[turn_idx]
            a2_labels   = agent2_labels_list[turn_idx]
            a2_enc_mask = agent2_encoder_attention_mask_list[turn_idx]
            a2_long_dec = agent2_long_decoder_input_ids_list[turn_idx]
            a2_long_lbl = agent2_long_labels_list[turn_idx]
            a2_model_pos = agent2_model_answer_position[:, turn_idx] - 1
            a2_ref_pos   = agent2_ref_answer_position[:, turn_idx] - 1

            a2_enc_embs = self.get_embd(self.codi2, self.model2_name)(a2_enc_ids)
            if turn_idx > 0 and agent1_gen_latents:
                a2_enc_embs = self._inject_other_agent_latent(
                    a2_enc_ids, a2_enc_embs,
                    agent1_gen_latents[turn_idx - 1],
                    self.adapter_1to2,
                    self.latent_id2,
                )

            if agent2_past_kv is not None:
                past_len = agent2_past_kv[0][0].size(2)
                past_mask = torch.ones(a2_enc_embs.size(0), past_len,
                                       dtype=a2_enc_mask.dtype, device=a2_enc_mask.device)
                a2_enc_mask = torch.cat([past_mask, a2_enc_mask], dim=1)

            out = self.codi2(inputs_embeds=a2_enc_embs, use_cache=True,
                             output_hidden_states=True, past_key_values=agent2_past_kv,
                             attention_mask=a2_enc_mask)
            agent2_past_kv = out.past_key_values
            latent_embd = out.hidden_states[-1][:, -1:, :]
            if self.use_prj:
                latent_embd = self.prj2(latent_embd)

            latent_list = []
            for _ in range(self.num_latent):
                out = self.codi2(inputs_embeds=latent_embd, use_cache=True,
                                 output_hidden_states=True, past_key_values=agent2_past_kv)
                agent2_past_kv = out.past_key_values
                latent_embd = out.hidden_states[-1][:, -1:, :]
                if self.use_prj:
                    latent_embd = self.prj2(latent_embd)
                latent_list.append(latent_embd)
            agent2_current_latents = torch.cat(latent_list, dim=1)  # [B, num_latent, H2]
            agent2_gen_latents.append(agent2_current_latents)

            a2_dec_embs = self.get_embd(self.codi2, self.model2_name)(a2_dec_ids)
            dyn_mask = None
            if self.fix_attn_mask:
                dyn_mask = torch.cat([
                    a2_enc_mask,
                    torch.ones(batch_size, self.num_latent, device=dev),
                    torch.ones(batch_size, a2_dec_embs.size(1), device=dev),
                ], dim=1).bool()

            out = self.codi2(inputs_embeds=a2_dec_embs, use_cache=True,
                             output_hidden_states=True, past_key_values=agent2_past_kv,
                             attention_mask=dyn_mask)
            agent2_past_kv = out.past_key_values

            # Agent2 CE loss
            logits = out.logits
            masked_labels = a2_labels.clone()
            masked_labels[~sample_should_compute] = IGNORE_INDEX
            valid = (masked_labels[:, 1:] != IGNORE_INDEX).sum()
            ce_loss = (
                self.loss_fct(logits[:, :-1].reshape(-1, logits.size(-1)),
                              masked_labels[:, 1:].reshape(-1))
                if valid > 0 else torch.tensor(0.0, device=dev).to(logits)
            )
            total_ce_loss = total_ce_loss + ce_loss

            # Agent2 distill loss (against its own teacher)
            distill_loss = torch.tensor(0.0, device=dev).to(ce_loss)
            for _, (student_hs, teacher_hs) in enumerate(
                    zip(out.hidden_states, agent2_ref_outputs_no_grad.hidden_states)):
                ref_sel = teacher_hs.gather(
                    1, a2_ref_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, teacher_hs.size(-1)))
                stu_sel = student_hs.gather(
                    1, a2_model_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, student_hs.size(-1)))
                if sample_should_compute.any():
                    dt = self.distill_loss_fct(stu_sel[sample_should_compute],
                                               ref_sel[sample_should_compute].detach())
                    if self.distill_loss_div_std:
                        std = ref_sel[sample_should_compute].std()
                        if std > 0:
                            dt = dt / std
                    distill_loss = distill_loss + dt
            distill_loss = distill_loss / len(out.hidden_states) * self.distill_loss_factor
            total_distill_loss = total_distill_loss + distill_loss

            decode_loss = torch.tensor(0, device=dev).to(ce_loss)
            decode_loss = decode_loss * self.decode_loss_factor
            total_decode_loss = total_decode_loss + decode_loss

            loss_detail["agent2_student_ce"].append(ce_loss.detach().item())
            loss_detail["agent2_distill"].append(distill_loss.detach().item())
            loss_detail["agent2_decode"].append(decode_loss.detach().item())
            loss_detail["agent2_total"].append((ce_loss + distill_loss + decode_loss).detach().item())

        loss = total_ce_loss + total_distill_loss + total_ref_loss + total_decode_loss

        if self.print_loss:
            print(f"total_ce={total_ce_loss:.4f}, distill={total_distill_loss:.4f}, "
                  f"ref={total_ref_loss:.4f}, decode={total_decode_loss:.4f}")

        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": total_ce_loss.detach().item() if torch.is_tensor(total_ce_loss) else total_ce_loss,
            "distill_loss": total_distill_loss.detach().item() if torch.is_tensor(total_distill_loss) else total_distill_loss,
            "ref_ce_loss": total_ref_loss.detach().item(),
            "decode_loss": total_decode_loss.detach().item() if torch.is_tensor(total_decode_loss) else total_decode_loss,
            "loss_detail": loss_detail,
        }
