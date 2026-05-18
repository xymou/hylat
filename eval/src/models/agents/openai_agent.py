import torch
from typing import List
from openai import OpenAI


class OpenAIAgent:
    def __init__(self,
        base_url,
        api_key,
        model_name,
        max_new_tokens,
        temperature,
        role_prompt):
        
        self.agent_type = 'openai'
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.role_prompt = role_prompt        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
    
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
    
    def generate(self, messages: List[str]) -> str:
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            max_tokens = self.max_new_tokens,
            temperature = self.temperature
        )
        response = response.choices[0].message.content
        # 在这里，生成完了直接记录
        self.assistant_output.append(response)
        self.history_msgs.append({
            "role": "assistant",
            "content": response
        })

        return response