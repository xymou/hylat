import torch
import torch.nn.functional as F
from typing import List
from src.models.agents.base_agent import BaseAgent


class HeteroLatentAgent(BaseAgent):
    """
    Latent agent backed by one sub-model of a heterogeneous dual-agent CODI
    (model_decoder_mas_dual_hetero.CODI).

    agent_idx=0 → codi1 (e.g. Llama)
    agent_idx=1 → codi2 (e.g. Qwen)

    Both agents share the same CODI model object. Cross-model latent
    projection is handled by HeteroLatentDebateTask via adapter_1to2 /
    adapter_2to1 before embedding concatenation — no <latent> placeholder
    injection needed at inference time.
    """

    def __init__(self, *args, agent_idx: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_idx  = agent_idx
        self.agent_type = 'hetero_latent'

        if agent_idx == 0:
            self.tokenizer   = self.model.tokenizer1
            self._codi       = self.model.codi1
            self._bot_id     = self.model.bot_id1
            self._eot_id     = self.model.eot_id1
            self._model_name = self.model.model1_name
        else:
            self.tokenizer   = self.model.tokenizer2
            self._codi       = self.model.codi2
            self._bot_id     = self.model.bot_id2
            self._eot_id     = self.model.eot_id2
            self._model_name = self.model.model2_name

        self.latent_num = self.model.num_latent
        self.use_prj    = self.model.use_prj
        self.init_chat_template()

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_prj(self):
        if not self.use_prj:
            return None
        return self.model.prj1 if self.agent_idx == 0 else self.model.prj2

    def _get_embd(self):
        return self.model.get_embd(self._codi, self._model_name)

    def init_chat_template(self):
        super().init_chat_template()
        self.user_embs_fr = (
            self.token_to_embedding(self.user_prompt_fr)
            if self.user_prompt_fr else None
        )
        self.user_embs_ed      = self.token_to_embedding(self.user_prompt_ed)
        self.assistant_embs_fr = self.token_to_embedding(self.assistant_prompt_fr)
        self.assistant_embs_ed = self.token_to_embedding(self.assistant_prompt_ed)

    def token_to_embedding(self, input_ids: List[int], del_bos_token: bool = False) -> torch.Tensor:
        if del_bos_token:
            input_ids = input_ids[1:]
        return self._get_embd()(
            torch.tensor(input_ids).to(self._codi.device)
        )

    def text_to_embedding(self, text: str, del_bos_token: bool = False) -> torch.Tensor:
        return self.token_to_embedding(
            input_ids=self.tokenizer.encode(text, add_special_tokens=False),
            del_bos_token=del_bos_token,
        )

    # ── state initialisation ─────────────────────────────────────────────

    def init_history(self, first_user_prompt: str):
        message = []
        self.assistant_output      = []
        self.assistant_output_text = []
        if self.role_prompt is not None:
            message.append({"role": "system", "content": self.role_prompt})
        message.append({"role": "user", "content": first_user_prompt})
        input_ids         = self.tokenizer.apply_chat_template(message, add_generation_prompt=True)
        self.history_embs = self.token_to_embedding(input_ids)
        self.history_msgs = message

    # ── generation ───────────────────────────────────────────────────────

    def generate(self, token_embeds: torch.Tensor) -> torch.Tensor:
        token_embeds = token_embeds.unsqueeze(0)
        device       = token_embeds.device
        embd_fn      = self._get_embd()
        prj          = self._get_prj()

        bot_embd  = embd_fn(
            torch.tensor([self._bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)
        first_input = torch.cat([token_embeds, bot_embd], dim=1)
        attn_mask   = torch.ones(first_input.shape[:2], device=device)

        past_kv = None
        with torch.no_grad():
            out = self._codi(
                inputs_embeds=first_input,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_kv,
                attention_mask=attn_mask,
            )
            past_kv     = out.past_key_values
            latent_embd = out.hidden_states[-1][:, -1, :].unsqueeze(1)
            if prj is not None:
                latent_embd = prj(latent_embd)
            latent_list = [bot_embd]

            for _ in range(self.latent_num):
                out = self._codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_kv,
                )
                past_kv     = out.past_key_values
                latent_embd = out.hidden_states[-1][:, -1, :].unsqueeze(1)
                if prj is not None:
                    latent_embd = prj(latent_embd)
                latent_list.append(latent_embd)

            eot_embd = embd_fn(
                torch.tensor([self._eot_id], dtype=torch.long, device=device)
            ).unsqueeze(0)
            latent_list.append(eot_embd)
            output      = eot_embd
            pred_tokens = []

            for _ in range(self.max_new_tokens):
                out = self._codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_kv,
                )
                past_kv = out.past_key_values
                logits  = out.logits[:, -1, :self._codi.config.vocab_size - 1]

                if not self.generation_configs or not self.generation_configs["temperature"]:
                    next_token_id = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits = logits / self.generation_configs["temperature"]
                    if self.generation_configs.get("top_k", 1) > 1:
                        top_k_vals, _ = torch.topk(logits, self.generation_configs["top_k"], dim=-1)
                        logits[logits < top_k_vals[:, -1:]] = -float("inf")
                    if self.generation_configs.get("top_p", 1.0) < 1.0:
                        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        remove    = cum_probs > self.generation_configs["top_p"]
                        remove    = remove.roll(1, dims=-1)
                        remove[:, 0] = False
                        for b in range(logits.size(0)):
                            logits[b, sorted_idx[b, remove[b]]] = -float("inf")
                    probs         = F.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

                pred_tokens.append(next_token_id.item())
                if next_token_id == self.tokenizer.eos_token_id:
                    break

                output = embd_fn(next_token_id).unsqueeze(0)
                if output.dim() == 2:
                    output = output.unsqueeze(0)
                latent_list.append(output)

        output_embed = torch.cat(latent_list, dim=1).squeeze(0)
        text         = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
        self.assistant_output.append(output_embed)
        self.assistant_output_text.append(text)

        self.history_embs = torch.cat([
            self.history_embs,
            output_embed,
            self.assistant_embs_ed,
        ], dim=0)
        self.history_msgs.append({"role": "assistant", "content": text})
        return output_embed
