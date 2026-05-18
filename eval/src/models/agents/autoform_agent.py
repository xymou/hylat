import torch
from typing import List

from src.models.agents.base_agent import BaseAgent


class AutoFormAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'autoform'
    
    def init_history(self, first_user_prompt):
        self.assistant_output = []
        if self.role_prompt is not None:
            self.history_msgs = [{
                "role": "system", 
                "content": self.role_prompt
            }]
        else:
            self.history_msgs = []
        self.history_msgs.append({
            "role": "user",
            "content": first_user_prompt
        })
    
    def generate(self, messages: List[str], add_assistant_prompt: str = None) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
        )
        if add_assistant_prompt is not None:
            input_ids += self.tokenizer.encode(add_assistant_prompt, add_special_tokens=False)
        input_len = len(input_ids)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)
        attention_mask = torch.ones(input_ids.shape).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens, 
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
                **self.generation_configs,
            )
        output = output[0][input_len:]
        text = self.tokenizer.decode(output, skip_special_tokens=True)

        # 在这里，生成完了直接记录
        self.assistant_output.append(text)
        self.history_msgs.append({
            "role": "assistant",
            "content": text
        })

        return text