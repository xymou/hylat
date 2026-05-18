import os
import yaml
import copy
import time
import torch
from typing import Union, List

from src.tasks.utils import (
    get_merged_embedding_for_cipher,
    get_merged_ids_mask_hs_for_sde,
    get_merged_embedding_for_latent
)

from src.tasks.base_task import BaseTask


class IATask(BaseTask):
    def __init__(self, agents, dataset, args):
        super().__init__(agents, dataset, args)
        with open(args.prompt_file) as fin:
            name = "mmlu" if args.dataset.startswith("mmlu_") else args.dataset
            self.prompt_template = yaml.safe_load(fin)[name]
    
    def run(self, data):
        question = data["question"]
        ground_truth = data["answer"]
        passages = data["passages"]
        detail = {
            "question": question, 
            "ground_truth": ground_truth,
            "passages": passages,
        }
        prompts = copy.deepcopy(self.prompt_template)
        for kk in prompts:
            prompts[kk] = prompts[kk].replace("{question}", question)
        
        # allocate private knowledge
        allocate_psg = []
        for _ in range(self.args.agent_cnt):
            allocate_psg.append([])
        for idx, psg in enumerate(passages):
            allocate_psg[idx % self.args.agent_cnt].append(psg)
        for agent, allo_psg in zip(self.agents, allocate_psg):
            segments = ""
            for idx, psg in enumerate(allo_psg):
                segments += f"Document {idx+1}: {psg}\n"
            agent.role_prompt = prompts["role_prompt"].replace("{segments}", segments)
            agent.private_knowledge = segments

        run_time, use_rounds = self.run_func(prompts)

        # evaluate
        for agent_idx, agent in enumerate(self.agents):
            det = self.dataset.evaluate(
                output=agent.final_output_text,
                test_id=data["test_id"], 
            )
            for met in det:
                if met in self.dataset.eval_metrics:
                    try:
                        det[met] = det[met].item()
                    except:
                        pass
            det["history"] = agent.history
            detail[f"agent_{agent_idx}"] = det
        detail["run_time(s)"] = run_time
        detail["use_rounds"] = use_rounds
        return detail
    
    def run_func(self, prompts):
        raise NotImplementedError
    
    def generate_result(self, details):
        res = {
            "run_time(s)": sum([d["run_time(s)"] for d in details]) / len(details),
            "use_rounds": sum([d["use_rounds"] for d in details]) / len(details),
        }
        agent_cnt = self.args.agent_cnt
        for agent_idx in range(agent_cnt):
            agent_res = {
                met: sum([float(d[f"agent_{agent_idx}"][met]) for d in details]) / len(details)
                for met in self.dataset.eval_metrics
            }
            res[f"agent_{agent_idx}"] = agent_res
        if self.args.method == "single":
            average = {}
            for met in self.dataset.eval_metrics:
                average[met] = max(
                    [float(res[f"agent_{agent_idx}"][met]) for agent_idx in range(agent_cnt)]
                )
        else:
            average = {}
            for met in self.dataset.eval_metrics:
                val = []
                for det in details:
                    marked = []
                    no_marked = []
                    for agent_idx in range(agent_cnt):
                        if det[f"agent_{agent_idx}"]["marked_answer"]:
                            marked.append(det[f"agent_{agent_idx}"][met])
                        else:
                            no_marked.append(det[f"agent_{agent_idx}"][met])
                    if len(marked) > 0:
                        val.append(sum(marked) / len(marked))
                    else:
                        val.append(sum(no_marked) / len(no_marked))
                average[met] = sum(val) / len(val)
        res["average"] = average
        return res


class SingleIATask(IATask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        for agent in self.agents:
            prompt = prompts["direct_prompt"].replace("{segments}", agent.private_knowledge)
            agent.role_prompt = None
            agent.init_history(first_user_prompt=prompt)
            output = agent.generate(agent.history_msgs)
            agent.final_output_text = output
            agent.history = agent.history_msgs
        end_time = time.perf_counter()
        return end_time - start_time, 1


class NlIATask(IATask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        use_rounds = -1
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            text = agent.generate(agent.history_msgs)
            if any([w in text for w in self.dataset.stop_words]):
                use_rounds = 1 
        
        for rd in range(self.args.rounds-1): # 第一轮独立出去了
            if use_rounds != -1:
                break
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                all_other_resp = ""
                for other in range(agent_cnt):
                    if other != cur:
                        other_resp = self.agents[other].assistant_output[rd]
                        all_other_resp += prompts["other_response_prompt"].replace("{other_response}", other_resp)
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["communication_prompt"].replace("{all_other_response}", all_other_resp)
                })
                text = agent.generate(agent.history_msgs)
                if any([w in text for w in self.dataset.stop_words]):
                    use_rounds = rd + 2

        for agent in self.agents:
            agent.final_output_text = agent.assistant_output[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time, use_rounds
        


class CipherIATask(IATask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        use_rounds = -1
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            output_embs = agent.generate(agent.history_embs)
            output_text = agent.get_human_output(output_embs)
            if any([w in output_text for w in self.dataset.stop_words]):
                use_rounds = 1
        
        for rd in range(self.args.rounds-1): # 第一轮独立出去了
            if use_rounds != -1:
                break
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                merged_other_embs = []
                for other in range(agent_cnt):
                    if cur != other:
                        merged_other_embs.append(
                            get_merged_embedding_for_cipher(
                                t2e_func=agent.text_to_embedding, 
                                prompt_template=prompts["other_response_prompt"],
                                placeholder="{other_response}",
                                input_embs=self.agents[other].assistant_output[rd]
                            )
                        )
                user_embs = get_merged_embedding_for_cipher(
                    t2e_func=agent.text_to_embedding,
                    prompt_template=prompts["communication_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=torch.cat(merged_other_embs, dim=0),
                )
                agent.history_embs = torch.cat([
                    agent.history_embs, 
                    agent.user_embs_fr, 
                    user_embs, 
                    agent.user_embs_ed, 
                    agent.assistant_embs_fr, 
                ], dim=0)
                output_embs = agent.generate(agent.history_embs)
                output_text = agent.get_human_output(output_embs)
                if any([w in output_text for w in self.dataset.stop_words]):
                    use_rounds = rd + 2
        
        for agent in self.agents:
            agent.final_output_text = agent.get_human_output(agent.assistant_output[-1])
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.get_human_output(agent.history_embs)
        return end_time - start_time, use_rounds
        


class SDEIATask(IATask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        use_rounds = -1
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            output_ids, _ = agent.generate(
                input_ids = agent.history_ids,
                if_edit=False,
                edit_layer_idx=self.args.edit_layer_idx,
            )
            output_text = agent.get_human_output(output_ids)
            if any([w in output_text for w in self.dataset.stop_words]):
                use_rounds = 1
        
        for rd in range(self.args.rounds-1): # 第一轮独立出去了
            if use_rounds != -1:
                break
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                # 获取其他模型的输出
                merged_input_ids = []
                merged_mask = []
                merged_hs = []
                for other in range(agent_cnt):
                    if other != cur:
                        resp_ids = self.agents[other].assistant_ids[rd]
                        ids, mask, hs = get_merged_ids_mask_hs_for_sde(
                            tokenizer=agent.tokenizer,
                            prompt_template=prompts["other_response_prompt"],
                            placeholder="{other_response}",
                            input_ids=resp_ids,  
                            input_mask=torch.zeros(len(resp_ids), dtype=torch.bool),
                            input_hs=self.agents[other].assistant_hs[rd],
                        )
                        merged_input_ids.append(ids)
                        merged_mask.append(mask)
                        merged_hs.append(hs)
                all_resp_ids = []
                for ids in merged_input_ids:
                    all_resp_ids += ids
                user_input_ids, user_mask, user_hs = get_merged_ids_mask_hs_for_sde(
                    tokenizer=agent.tokenizer,
                    prompt_template=prompts["communication_prompt"],
                    placeholder="{all_other_response}",
                    input_ids=all_resp_ids,
                    input_mask=torch.cat(merged_mask, dim=0),
                    input_hs={
                        layer_idx: torch.cat([merged_hs[_][layer_idx] for _ in range(len(merged_hs))], dim=1) 
                        for layer_idx in self.args.edit_layer_idx
                    },
                )
                history_mask = torch.cat([
                    torch.zeros(len(agent.history_ids) + len(agent.user_prompt_fr), dtype=torch.bool),
                    user_mask,
                    torch.zeros(len(agent.user_prompt_ed) + len(agent.assistant_prompt_fr), dtype=torch.bool),
                ], dim=0)
                history_hs = {}
                for layer_idx, hs in user_hs.items():
                    history_hs[layer_idx] = torch.cat([
                        torch.zeros((1, len(agent.history_ids)+len(agent.user_prompt_fr), hs.shape[-1])),
                        hs,
                        torch.zeros((1, len(agent.user_prompt_ed)+len(agent.assistant_prompt_fr), hs.shape[-1])),
                    ], dim=1)
                agent.history_ids = agent.history_ids + agent.user_prompt_fr + \
                    user_input_ids + agent.user_prompt_ed + agent.assistant_prompt_fr
                
                output_ids, _ = agent.generate(
                    input_ids=agent.history_ids, 
                    if_edit=True, 
                    edit_layer_idx=self.args.edit_layer_idx,
                    edit_mask=history_mask,
                    edit_tensor=history_hs,
                )
                output_text = agent.get_human_output(output_ids)
                if any([w in output_text for w in self.dataset.stop_words]):
                    use_rounds = rd + 2
        
        for agent in self.agents:
            agent.final_output_text = agent.get_human_output(agent.assistant_ids[-1])
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.get_human_output(agent.history_ids)
        return end_time - start_time, use_rounds
    
    
class LatentStage1IATask(IATask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        use_rounds = -1
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            text = agent.generate(agent.history_embs)
            if any([w in text for w in self.dataset.stop_words]):
                use_rounds = 1 
        
        for rd in range(self.args.rounds-1):
            if use_rounds != -1:
                break
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                all_other_resp = ""
                for other in range(agent_cnt):
                    if other != cur:
                        other_resp = self.agents[other].assistant_output[rd]
                        all_other_resp += prompts["other_response_prompt"].replace("{other_response}", other_resp)
                user_embs = agent.text_to_embedding(all_other_resp)
                user_embs = get_merged_embedding_for_latent(
                    t2e_func=agent.text_to_embedding,
                    prompt_template=prompts["communication_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=user_embs,
                )
                agent.history_embs = torch.cat([
                    agent.history_embs, 
                    agent.user_embs_fr,
                    user_embs, 
                    agent.user_embs_ed, 
                    agent.assistant_embs_fr,                     
                ], dim=0)
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["communication_prompt"].replace("{all_other_response}", all_other_resp)
                })
                text = agent.generate(agent.history_embs)
                if any([w in text for w in self.dataset.stop_words]):
                    use_rounds = rd + 2

        for agent in self.agents:
            agent.final_output_text = agent.assistant_output[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time, use_rounds
    
    
class LatentStage2IATask(IATask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        use_rounds = -1
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            output_embs = agent.generate(agent.history_embs)
            output_text = agent.assistant_output_text[-1]
            if any([w in output_text for w in self.dataset.stop_words]):
                use_rounds = 1
        
        for rd in range(self.args.rounds-1):
            if use_rounds != -1:
                break
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                merged_other_embs = []
                all_other_resp = ""
                for other in range(agent_cnt):
                    if cur != other:
                        other_response = get_merged_embedding_for_latent(
                            t2e_func=agent.text_to_embedding, 
                            prompt_template=prompts["other_response_prompt"],
                            placeholder="{other_response}",
                            input_embs=self.agents[other].assistant_output[rd]                            
                        )
                        merged_other_embs.append(other_response)
                        other_resp = self.agents[other].assistant_output_text[rd]
                        all_other_resp += prompts["other_response_prompt"].replace("{other_response}", other_resp)
                        
                user_embs = get_merged_embedding_for_latent(
                    t2e_func=agent.text_to_embedding,
                    prompt_template=prompts["communication_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=torch.cat(merged_other_embs, dim=0),                    
                )
                agent.history_embs = torch.cat([
                    agent.history_embs, 
                    agent.user_embs_fr, 
                    user_embs, 
                    agent.user_embs_ed, 
                    agent.assistant_embs_fr, 
                ], dim=0)
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["communication_prompt"].replace("{all_other_response}", all_other_resp)
                })    
                output_embs = agent.generate(agent.history_embs)    
                output_text = agent.assistant_output_text[-1]
                if any([w in output_text for w in self.dataset.stop_words]):
                    use_rounds = rd + 2
        
        for agent in self.agents:
            agent.final_output_text = agent.assistant_output_text[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time, use_rounds