"""
stage2: agents generate latent+text, and provide latent+text to others
"""
import torch
import torch.nn.functional as F
from typing import List
from src.models.agents.base_agent import BaseAgent


class LatentStage2Agent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'latent2'
        self.latent_num = self.model.num_latent
        self.use_prj = self.model.use_prj
        self.init_chat_template()
        self.model.to(torch.float32)
    
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
        return self.model.get_embd(self.model.codi, self.model.model_name)(torch.tensor(input_ids).to(self.model.codi.device))    

    def text_to_embedding(self, text: str, del_bos_token: bool=False) -> List[torch.Tensor]:
        return self.token_to_embedding(
            input_ids=self.tokenizer.encode(text, add_special_tokens=False), 
            del_bos_token=del_bos_token,
        )
   
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
    
    def generate(self, token_embeds: torch.Tensor):
        batch_size = 1
        token_embeds = token_embeds.unsqueeze(0)
        device = token_embeds.device
        bot_embd = self.model.get_embd(self.model.codi, self.model.model_name)(torch.tensor([self.model.bot_id], dtype=torch.long, device=device)).unsqueeze(0).to(device)   
        first_input_embeds = torch.cat([token_embeds, bot_embd], dim=1)
        attention_mask = torch.ones((first_input_embeds.size()[:2])).to(first_input_embeds.device)
        
        past_key_values = None
        with torch.no_grad():
            outputs = self.model.codi(
                inputs_embeds=first_input_embeds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask = attention_mask
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)  # bot 位置的 hidden state
        
            if self.use_prj:
                latent_embd = self.model.prj(latent_embd)
            latent_list = [bot_embd]
        
            # generate latent
            for i in range(self.latent_num):
                outputs = self.model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                if self.use_prj:
                    latent_embd = self.model.prj(latent_embd)

                latent_list.append(latent_embd)
            
                     
            eot_embd = self.model.get_embd(self.model.codi, self.model.model_name)(torch.tensor([self.model.eot_id], dtype=torch.long, device=device)).unsqueeze(0)
            output = eot_embd
            latent_list.append(eot_embd)
            
            seq_len = 0
            pred_tokens = [] 
            
            for i in range(self.max_new_tokens):
                seq_len += 1             
                out = self.model.codi(
                        inputs_embeds=output,
                        output_hidden_states=False,
                        attention_mask=None,
                        use_cache=True,
                        output_attentions=False,
                        past_key_values=past_key_values,
                    )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :self.model.codi.config.vocab_size-1]

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
            
                # handle EOS
                pred_tokens.append(next_token_ids.item())
                if next_token_ids == self.model.tokenizer.eos_token_id:
                    break
                
                # output = self.model.get_embd(self.model.codi, self.model.model_name)(next_token_ids).unsqueeze(0).unsqueeze(0).to(device)
                output = self.model.get_embd(self.model.codi, self.model.model_name)(next_token_ids).unsqueeze(0).to(device)
                if len(output.size())==2:
                    output = output.unsqueeze(0) 
                latent_list.append(output)
        
        output_embed = torch.cat(latent_list, dim=1).squeeze(0)
        text = self.model.tokenizer.decode(pred_tokens, skip_special_tokens=True)
        self.assistant_output.append(output_embed)
        self.assistant_output_text.append(text)
        
        self.history_embs = torch.cat([
            self.history_embs,
            output_embed,
            self.assistant_embs_ed
        ], dim=0)
        self.history_msgs.append({
            "role": "assistant",
            "content": text
        })
        return output_embed
    
    
class PureLatentAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'latent2'
        self.latent_num = self.model.num_latent
        self.use_prj = self.model.use_prj
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
    
    def token_to_embedding(self, input_ids: List[int], del_bos_token: bool=False) -> List[torch.Tensor]:
        if del_bos_token:
            input_ids = input_ids[1:]
        return self.model.get_embd(self.model.codi, self.model.model_name)(torch.tensor(input_ids).to(self.model.codi.device))    

    def text_to_embedding(self, text: str, del_bos_token: bool=False) -> List[torch.Tensor]:
        return self.token_to_embedding(
            input_ids=self.tokenizer.encode(text, add_special_tokens=False), 
            del_bos_token=del_bos_token,
        )
   
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
    
    def generate(self, token_embeds: torch.Tensor, generate_text=False):
        batch_size = 1
        token_embeds = token_embeds.unsqueeze(0)
        device = token_embeds.device
        bot_embd = self.model.get_embd(self.model.codi, self.model.model_name)(torch.tensor([self.model.bot_id], dtype=torch.long, device=device)).unsqueeze(0).to(device)   
        first_input_embeds = torch.cat([token_embeds, bot_embd], dim=1)
        attention_mask = torch.ones((first_input_embeds.size()[:2])).to(first_input_embeds.device)
        
        past_key_values = None
        with torch.no_grad():
            outputs = self.model.codi(
                inputs_embeds=first_input_embeds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask = attention_mask
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)  # bot 位置的 hidden state
        
            if self.use_prj:
                latent_embd = self.model.prj(latent_embd)
            latent_list = [bot_embd]
        
            # generate latent
            for i in range(self.latent_num):
                outputs = self.model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                if self.use_prj:
                    latent_embd = self.model.prj(latent_embd)
                latent_list.append(latent_embd)
                
            eot_embd = self.model.get_embd(self.model.codi, self.model.model_name)(torch.tensor([self.model.eot_id], dtype=torch.long, device=device)).unsqueeze(0)
            output = eot_embd
            latent_list.append(eot_embd)
            
            seq_len = 0
            pred_tokens = [] 
            
            if generate_text:
                for i in range(self.max_new_tokens):
                    seq_len += 1               
                    out = self.model.codi(
                            inputs_embeds=output,
                            output_hidden_states=False,
                            attention_mask=None,
                            use_cache=True,
                            output_attentions=False,
                            past_key_values=past_key_values
                        )
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :self.model.codi.config.vocab_size-1]
                    
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
                
                    # handle EOS

                    pred_tokens.append(next_token_ids.item())
                    if next_token_ids == self.model.tokenizer.eos_token_id:
                        break
                    
                    # output = self.model.get_embd(self.model.codi, self.model.model_name)(next_token_ids).unsqueeze(0).unsqueeze(0).to(device)
                    output = self.model.get_embd(self.model.codi, self.model.model_name)(next_token_ids).unsqueeze(0).to(device)
                    if len(output.size())==2:
                        output = output.unsqueeze(0) 
                    latent_list.append(output)
        
        output_embed = torch.cat(latent_list, dim=1).squeeze(0)
        text = self.model.tokenizer.decode(pred_tokens, skip_special_tokens=True) if pred_tokens else ""
        self.assistant_output.append(output_embed)
        self.assistant_output_text.append(text)
        
        self.history_embs = torch.cat([
            self.history_embs,
            output_embed,
            self.assistant_embs_ed
        ], dim=0)
        self.history_msgs.append({
            "role": "assistant",
            "content": text
        })
        return output_embed