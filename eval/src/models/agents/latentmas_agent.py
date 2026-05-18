"""
LatentMAS agent
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from src.models.agents.base_agent import BaseAgent

def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


class LatentMASAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.latent_steps = kwargs["latent_steps"]
        self.args = kwargs["latentmas_args"]
        del kwargs["latent_steps"]
        del kwargs["latentmas_args"]
        super().__init__(*args, **kwargs)
        self.agent_type = 'latentmas'
        self.emb_table = self.model.model.embed_tokens
        self.init_chat_template()
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.device = self.model.device
    
    def init_chat_template(self):
        super().init_chat_template()
        if self.user_prompt_fr:
            self.user_embs_fr = self.token_to_embedding(self.user_prompt_fr)
        else:
            self.user_embs_fr = None
        self.user_embs_ed = self.token_to_embedding(self.user_prompt_ed)
        self.assistant_embs_fr = self.token_to_embedding(self.assistant_prompt_fr)
        self.assistant_embs_ed = self.token_to_embedding(self.assistant_prompt_ed)
    
    def token_to_embedding(self, input_ids: List[int], del_bos_token: bool=False) -> List[torch.Tensor]:
        if del_bos_token:
            input_ids = input_ids[1:]
        return self.emb_table(torch.tensor(input_ids).to(self.model.device))
    
    def text_to_embedding(self, text: str, del_bos_token: bool=False) -> List[torch.Tensor]:
        return self.token_to_embedding(
            input_ids=self.tokenizer.encode(text, add_special_tokens=False), 
            del_bos_token=del_bos_token,
        )
   
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)   
   
   
    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor = None,
        input_embds: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None
        ):
        if input_ids is None and input_embds is None:
            raise ValueError("must provide either input_ids or input_embeds")
        if input_embds is not None:
            if len(input_embds.size())==2:
                input_embds = input_embds.unsqueeze(0)
        if attention_mask is None:
            if input_ids:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = torch.ones_like(input_embds[:,:,0], device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None        
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        if input_ids is not None:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **self.generation_configs,
            )     
            sequences = outputs.sequences
            generations: List[str] = []
            for idx, length in enumerate(prompt_lengths):
                length = int(length)
                generated_ids = sequences[idx, length:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                generations.append(text)
        else:
            pred_tokens = []
            output = input_embds
            for i in range(self.max_new_tokens):
                if len(output.size())==2:
                    output = output.unsqueeze(0)
                out = self.model(
                        inputs_embeds=output,
                        output_hidden_states=False,
                        attention_mask=None,
                        use_cache=True,
                        output_attentions=False,
                        past_key_values=past_key_values
                    )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :self.model.config.vocab_size-1]
                
                # implement the sampling process
                if not self.generation_configs or not self.generation_configs["temperature"]: # greedy
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= self.generation_configs["temperature"]
                    if self.generation_configs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, self.generation_configs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")
                    
                    if self.generation_configs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > self.generation_configs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                        
                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                pred_tokens.append(next_token_ids.item())
                if next_token_ids == self.tokenizer.eos_token_id:
                    break    
                output = self.token_to_embedding(next_token_ids).unsqueeze(0).to(self.device)
                             
            generations = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
        
        return generations
    
    
    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor = None,
        input_embds: torch.Tensor =  None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids is None and input_embds is None:
            raise ValueError("must provide either input_ids or input_embeds")

        if input_embds is not None:
            if len(input_embds.size())==2:
                input_embds = input_embds.unsqueeze(0)
        if attention_mask is None:
            if input_ids:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = torch.ones_like(input_embds[:,:,0], device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)
            
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        if input_ids is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = self.model(
                inputs_embeds=input_embds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )            
        past = outputs.past_key_values
        
        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())    
        
        for step in range(latent_steps):
            source_model = self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            
            latent_vecs_all.append(latent_vec.detach().clone())
            
            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )    
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        latent = torch.cat(latent_vecs_all, dim=0)
        return latent

 
    def init_history(self, first_user_prompt):
        message = []
        self.assistant_output = []
        self.assistant_output_text = []
        if self.role_prompt is not None:
            message.append({
                "role": "system", 
                "content": self.role_prompt,
            }) 
        message.append({"role": "user", "content": first_user_prompt})
        input_ids = self.tokenizer.apply_chat_template(message, add_generation_prompt=True)
        self.history_embs = self.token_to_embedding(input_ids)         
        self.history_msgs = message
    
    
    def generate(self, token_embeds: torch.Tensor, final=False):
        if not final:
            latent = self.generate_latent_batch(input_embds=token_embeds, latent_steps=self.latent_steps)
            self.assistant_output.append(latent)
            self.assistant_output_text.append("<latent>") # slot
            self.history_embs = torch.cat([
                self.history_embs,
                latent,
                self.assistant_embs_ed
            ], dim=0)
            self.history_msgs.append({
                "role": "assistant",
                "content": "<latent>"
            })            
        else:
            final_text = self.generate_text_batch(
                input_embds=token_embeds
            )
            self.assistant_output_text.append(final_text)
            self.history_msgs.append({
                "role": "assistant",
                "content": final_text
            })            
        return latent if not final else final_text
        
