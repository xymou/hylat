from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseAgent:
    def __init__(
        self, 
        engine_model_name_or_path: str, 
        engine_tokenizer: AutoTokenizer, 
        engine_model: AutoModelForCausalLM,
        generation_configs: dict, 
        max_new_tokens: int,
        role_prompt: str = None,
        agent_persona: str = None,
        agent_party: str = None
    ):
        self.model_name_or_path = engine_model_name_or_path
        self.tokenizer = engine_tokenizer
        self.model = engine_model
        self.generation_configs = generation_configs
        self.max_new_tokens = max_new_tokens
        self.role_prompt = role_prompt
        self.agent_persona = str(agent_persona)
        self.agent_party = agent_party


    def init_chat_template(self): 
        # This function is used to initialize the chat template for the tokenizer.
        # user
        msg1 = [{"role": "system", "content": ""}]
        msg2 = msg1 + [{"role": "user", "content": ""}]
        tk1 = self.tokenizer.apply_chat_template(msg1, add_generation_prompt=False)
        tk2 = self.tokenizer.apply_chat_template(msg2, add_generation_prompt=False)
        msg3 = msg1 + [{"role": "user", "content": "a"}]
        tk3 = self.tokenizer.apply_chat_template(msg3, add_generation_prompt=False)
        # the common suffix of tk2 and tk3 is the end of user prompt
        user_prompt_ed = None
        for i in range(len(tk2)):
            if tk2[-i-1] != tk3[-i-1]:
                user_prompt_ed = tk2[-i:]
                break
        # tk2 - tk1, and delete user_prompt_ed, is the user prompt's start
        user_prompt_fr = tk2[len(tk1):-len(user_prompt_ed)]

        # assitant
        tk4 = self.tokenizer.apply_chat_template(msg2, add_generation_prompt=True)
        # tk4 - tk2 is the assistant prompt's start
        assistant_prompt_fr = tk4[len(tk2):]
        msg4 = msg2 + [{"role": "assistant", "content": ""}]
        tk5 = self.tokenizer.apply_chat_template(msg4, add_generation_prompt=False)
        # tk5 - tk4 is the assistant prompt's end
        assistant_prompt_ed = tk5[len(tk4):]
        self.user_prompt_fr = user_prompt_fr
        self.user_prompt_ed = user_prompt_ed
        self.assistant_prompt_fr = assistant_prompt_fr
        self.assistant_prompt_ed = assistant_prompt_ed