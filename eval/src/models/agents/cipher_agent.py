import torch
from typing import List

from src.models.agents.base_agent import BaseAgent


class CipherAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_table = self.model.model.embed_tokens
        self.emb_table_norm = torch.nn.functional.normalize(self.emb_table.weight, p=2, dim=-1)
        self.agent_type = "cipher"
        self.init_chat_template()

    def init_chat_template(self):
        super().init_chat_template()
        if self.user_prompt_fr:
            self.user_embs_fr = self.token_to_embedding(self.user_prompt_fr)
        else:
            self.user_embs_fr = None
        self.user_embs_ed = self.token_to_embedding(self.user_prompt_ed)
        self.assistant_embs_fr = self.token_to_embedding(self.assistant_prompt_fr)
        self.assistant_embs_ed = self.token_to_embedding(self.assistant_prompt_ed)

    def init_history(self, first_user_prompt):
        message = []
        self.assistant_output = []
        if self.role_prompt is not None:
            message.append({
                "role": "system", 
                "content": self.role_prompt,
            }) 
        message.append({"role": "user", "content": first_user_prompt})
        input_ids = self.tokenizer.apply_chat_template(message, add_generation_prompt=True)
        self.history_embs = self.token_to_embedding(input_ids) 

    def token_to_embedding(self, input_ids: List[int], del_bos_token: bool=False) -> List[torch.Tensor]:
        if del_bos_token:
            input_ids = input_ids[1:]
        return self.emb_table(torch.tensor(input_ids).to(self.model.device))
    
    def text_to_embedding(self, text: str, del_bos_token: bool=False) -> List[torch.Tensor]:
        return self.token_to_embedding(
            input_ids=self.tokenizer.encode(text, add_special_tokens=False), 
            del_bos_token=del_bos_token,
        )
   
    def generate(self, token_embeds: torch.Tensor):
        temperature = self.generation_configs["temperature"]
        if temperature is None:
            temperature = self.model.generation_config.temperature
        token_embeds = token_embeds.unsqueeze(0) # shape add bsz
        input_len = token_embeds.shape[1]
        prev_pos = 0
        past_key_values = None
        output_embs = []
        for cur_pos in range(input_len, input_len + self.max_new_tokens):
            position_ids = torch.arange(prev_pos, cur_pos).long().unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                output = self.model.forward(
                    input_ids=None, 
                    inputs_embeds=token_embeds, 
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits / temperature if temperature > 0 else logits, dim=-1)
            next_token_emb = torch.einsum(
                "b v , v d -> b d", probs, self.emb_table.weight
            )  ## b: bsz = 1, v: vocab_size, d: hid_dim
            output_embs.append(next_token_emb)
            token_embeds = next_token_emb.unsqueeze(0)
            prev_pos = cur_pos
            next_token = torch.argmax(logits, dim=-1)
            if next_token[0] == self.tokenizer.eos_token_id:
                break

        output = torch.cat(output_embs, dim=0)
        self.assistant_output.append(output)
        self.history_embs = torch.cat([
            self.history_embs, 
            output, 
            self.assistant_embs_ed,
        ], dim=0)

        return output
    
    def truncate_output_embeddings(self, tokens: List[int]) -> List[int]:
        return tokens

    def get_human_output(self, embs: torch.Tensor) -> str:
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        dist = torch.cdist(embs, self.emb_table_norm, p=2)
        tokens = torch.argmin(dist, dim=-1).tolist() # 得到长度为 len 的 list
        tokens = self.truncate_output_embeddings(tokens)
        return self.tokenizer.decode(tokens)

    def get_token_ids(self, embs: torch.Tensor) -> str:
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        dist = torch.cdist(embs, self.emb_table_norm, p=2)
        tokens = torch.argmin(dist, dim=-1).tolist() # 得到长度为 len 的 list
        tokens = self.truncate_output_embeddings(tokens)
        return tokens