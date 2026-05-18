import torch
from typing import List, Optional, Callable, Union, Dict
from transformers import AutoTokenizer, GenerationConfig

from src.models.agents.base_agent import BaseAgent
from src.models.hs_edit_models import HiddenStatesEditingMethod, EditQwen2ForCausalLM, EditLlamaForCausalLM


class SDEAgent(BaseAgent):
    def __init__(
        self, 
        engine_model_name_or_path: str, 
        engine_tokenizer: AutoTokenizer, 
        engine_model: Union[EditQwen2ForCausalLM, EditLlamaForCausalLM],
        generation_configs: dict, 
        max_new_tokens: int,
        role_prompt: str = None,
        agent_persona: str = None,
        agent_party: str = None        
    ):
        self.model_name = engine_model_name_or_path
        self.tokenizer = engine_tokenizer
        self.model = engine_model
        self.generation_configs = generation_configs
        self.max_new_tokens = max_new_tokens
        self.role_prompt = role_prompt
        self.agent_persona = str(agent_persona)
        self.agent_party = agent_party

        self.agent_type = 'sde'
        self.init_chat_template()
    
    def init_history(self, first_user_prompt: str):
        messages = []
        self.assistant_ids = []
        self.assistant_hs = []

        if self.role_prompt is not None:
            messages.append({
                "role": "system",
                "content": self.role_prompt,
            })
        messages.append({
            "role": "user",
            "content": first_user_prompt,
        })
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        self.history_ids = input_ids
    
    def generate(
        self, 
        input_ids: List[int], 
        if_edit: bool=False,
        edit_layer_idx: Optional[List[int]]=None,
        edit_mask: Optional[torch.BoolTensor]=None,
        edit_tensor: Optional[Dict[int, torch.FloatTensor]]=None,
    ):
        input_len = len(input_ids)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)
        attention_mask = torch.ones(input_ids.shape).to(self.model.device)

        edit_method = HiddenStatesEditingMethod(
            if_edit=if_edit,
            edit_layer_idx=edit_layer_idx,
            edit_mask=edit_mask,
            edit_tensor=edit_tensor,
        )

        with torch.no_grad():
            if "pad_token_id" in self.generation_configs:
                output = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens+1, # 多生成一次，为了得到最后一个 token 之后的 hs
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    edit_method=edit_method,
                    **self.generation_configs,
                )
            else:
                output = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens+1, # 多生成一次，为了得到最后一个 token 之后的 hs
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    edit_method=edit_method,
                    **self.generation_configs,
                )
        output_ids = output.sequences[0][input_len:-1].tolist() # 相应的，这里也去掉最后一个生成的 token <ed>
        
        output_hs = {}
        # edit_method.saved_hs: List[Tensor] (bs, seq_len=1, hidden_size), len=len(output_ids)+1
        for layer_idx in edit_layer_idx:
            output_hs[layer_idx] = []
            saved = edit_method.saved_hs[layer_idx]
            assert len(saved) == len(output_ids) + 1 # 0 for input, i for token i
            for i in range(1, len(saved)):
                output_hs[layer_idx].append(saved[i] - saved[i-1])
            if len(output_hs[layer_idx]) > 0:
                output_hs[layer_idx] = torch.stack(output_hs[layer_idx], dim=1) # (bs, seq_len=len(output_ids), hidden_size)
            else:
                output_hs[layer_idx] = None

        self.assistant_ids.append(output_ids)
        self.assistant_hs.append(output_hs)
        self.history_ids = self.history_ids + output_ids + self.assistant_prompt_ed

        return output_ids, output_hs

    def get_human_output(self, input_ids: List[int]):
        return self.tokenizer.decode(input_ids)