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
from collections import Counter

class DebateTask(BaseTask):
    def __init__(self, agents, dataset, args):
        super().__init__(agents, dataset, args)
        if args.method == "single":
            if len(self.agents) > 0:
                self.agents = [self.agents[0]]
            self.args.agent_cnt = 1
        with open(args.prompt_file) as fin:
            # name = "mmlu" if args.dataset.startswith("mmlu_") else args.dataset
            if args.dataset.startswith("mmlu_"):
                name = "mmlu"
            elif args.dataset.startswith("arc_"):
                name = "arc"
            elif args.dataset.startswith("debug_"):
                name = "debug"
            else:
                name = args.dataset
            self.prompt_template = yaml.safe_load(fin)[name]
    
    def run(self, data):
        question = data["question"]
        ground_truth = data["answer"] 
        detail = {
            "question": question, 
            "ground_truth": ground_truth,
        }
        prompts = copy.deepcopy(self.prompt_template)
        for kk in prompts:
            prompts[kk] = prompts[kk].replace("{question}", question)
        
        run_time = self.run_func(prompts)
        major_ans = []

        for agent_idx, agent in enumerate(self.agents):
            det = self.dataset.evaluate(
                output=agent.final_output_text,
                test_id=data["test_id"],
            )
            major_ans.append(agent.final_output_text)
            for met in det:
                if met in self.dataset.eval_metrics:
                    try:
                        det[met] = det[met].item()
                    except:
                        pass
            det["history"] = agent.history
            detail[f"agent_{agent_idx}"] = det
        # majority vote
        major_det = self.dataset.evaluate_majority(
            outputs=major_ans,
            test_id=data["test_id"],
        )
        for met in major_det:
            if met in self.dataset.eval_metrics:
                try:
                    major_det[met] = major_det[met].item()
                except:
                    pass
        detail["majority_vote"] = major_det          
        
        detail["run_time(s)"] = run_time
        return detail
    
    def run_func(self, prompts):
        raise NotImplementedError

    def generate_result(self, details):
        res = {
            "run_time(s)": sum([d["run_time(s)"] for d in details]) / len(details),
        }
        agent_cnt = self.args.agent_cnt
        for agent_idx in range(agent_cnt):
            agent_res = {
                met: sum([float(d[f"agent_{agent_idx}"][met]) for d in details]) / len(details)
                for met in self.dataset.eval_metrics
            }
            res[f"agent_{agent_idx}"] = agent_res
        res["average"] = {
            met: sum([float(res[f"agent_{agent_idx}"][met]) for agent_idx in range(agent_cnt)]) / agent_cnt
            for met in self.dataset.eval_metrics
        }
        res["majority"] = {
            met: sum([float(d["majority_vote"][met]) for d in details]) / len(details)
            for met in self.dataset.eval_metrics
        }
        return res


class SingleDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent = self.agents[0]
        prompt = prompts["first_prompt"]
        agent.init_history(first_user_prompt=prompt)
        output = agent.generate(agent.history_msgs)
        agent.final_output_text = output
        agent.history = agent.history_msgs
        end_time = time.perf_counter()
        return end_time - start_time


class NlDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_msgs)
        
        for rd in range(self.args.rounds-1): # 第一轮独立出去了
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                all_other_resp = ""
                for other in range(agent_cnt):
                    if other != cur:
                        other_resp = self.agents[other].assistant_output[rd]
                        all_other_resp += prompts["other_response_prompt"].replace("{other_response}", other_resp)
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["debate_prompt"].replace("{all_other_response}", all_other_resp)
                })
                agent.generate(agent.history_msgs)
        for agent in self.agents:
            agent.final_output_text = agent.assistant_output[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time


class CipherDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_embs)
        
        for rd in range(self.args.rounds-1): # 第一轮独立出去了
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
                    prompt_template=prompts["debate_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=torch.cat(merged_other_embs, dim=0),
                )
                if agent.user_embs_fr is not None:
                    agent.history_embs = torch.cat([
                        agent.history_embs, 
                        agent.user_embs_fr, 
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr, 
                    ], dim=0)
                else:
                    agent.history_embs = torch.cat([
                        agent.history_embs,
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr, 
                    ], dim=0)                    
                agent.generate(agent.history_embs)
        for agent in self.agents:
            agent.final_output_text = agent.get_human_output(agent.assistant_output[-1])
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.get_human_output(agent.history_embs)
        return end_time - start_time


class SDEDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(
                input_ids = agent.history_ids,
                if_edit=False,
                edit_layer_idx=self.args.edit_layer_idx,
            )
        
        for rd in range(self.args.rounds-1): # 第一轮独立出去了
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
                    prompt_template=prompts["debate_prompt"],
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
                agent.generate(
                    input_ids=agent.history_ids, 
                    if_edit=True, 
                    edit_layer_idx=self.args.edit_layer_idx,
                    edit_mask=history_mask,
                    edit_tensor=history_hs,
                )
        for agent in self.agents:
            agent.final_output_text = agent.get_human_output(agent.assistant_ids[-1])
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.get_human_output(agent.history_ids)
        return end_time - start_time
    
class LatentStage1DebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_embs)
            
        for rd in range(self.args.rounds-1):
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                all_other_resp = ""
                for other in range(agent_cnt):
                    if cur != other:
                        other_resp = self.agents[other].assistant_output[rd]
                        all_other_resp += prompts["other_response_prompt"].replace("{other_response}", other_resp)
                user_embs = agent.text_to_embedding(all_other_resp)
                user_embs = get_merged_embedding_for_latent(
                    t2e_func=agent.text_to_embedding,
                    prompt_template=prompts["debate_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=user_embs,
                )
                if agent.user_embs_fr is not None:              
                    agent.history_embs = torch.cat([
                        agent.history_embs, 
                        agent.user_embs_fr,
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr,                     
                    ], dim=0)
                else:
                    agent.history_embs = torch.cat([
                        agent.history_embs, 
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr,                     
                    ], dim=0)
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["debate_prompt"].replace("{all_other_response}", all_other_resp)
                })
                agent.generate(agent.history_embs)
        for agent in self.agents:
            agent.final_output_text = agent.assistant_output[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time
    


class LatentStage2DebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt

        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_embs)
        
        for rd in range(self.args.rounds-1):
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
                    prompt_template=prompts["debate_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=torch.cat(merged_other_embs, dim=0),                    
                )
                if agent.user_embs_fr is not None:
                    agent.history_embs = torch.cat([
                        agent.history_embs, 
                        agent.user_embs_fr, 
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr, 
                    ], dim=0)
                else:
                    agent.history_embs = torch.cat([
                        agent.history_embs,
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr, 
                    ], dim=0)
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["debate_prompt"].replace("{all_other_response}", all_other_resp)
                })
                agent.generate(agent.history_embs)
        for agent in self.agents:
            agent.final_output_text = agent.assistant_output_text[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time                 
    

class ShortAnswerDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_msgs)
            
        for rd in range(self.args.rounds-1):
            for cur in range(agent_cnt):
                agent = self.agents[cur]
                all_other_resp = ""
                for other in range(agent_cnt):
                    if cur != other:
                        other_resp = self.agents[other].assistant_output[rd]
                        all_other_resp += prompts["other_response_prompt"].replace("{other_response}", other_resp)
    
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["debate_prompt"].replace("{all_other_response}", all_other_resp)
                })
                agent.generate(agent.history_msgs)
        for agent in self.agents:
            agent.final_output_text = agent.assistant_output[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time
    
    

class LatentMASDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
        # 首次独立出去
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_embs)
        
        for rd in range(self.args.rounds-1):
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
                    prompt_template=prompts["debate_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=torch.cat(merged_other_embs, dim=0),                    
                )
                if agent.user_embs_fr is not None:
                    agent.history_embs = torch.cat([
                        agent.history_embs, 
                        agent.user_embs_fr, 
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr, 
                    ], dim=0)
                else:
                    agent.history_embs = torch.cat([
                        agent.history_embs, 
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr, 
                    ], dim=0)
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["debate_prompt"].replace("{all_other_response}", all_other_resp)
                })
                if rd == self.args.rounds-2:
                    agent.generate(agent.history_embs, final=True)
                else:
                    agent.generate(agent.history_embs, final=False)
        for agent in self.agents:
            agent.final_output_text = agent.assistant_output_text[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time 
    
    
class PureLatentDebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt

        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_embs, generate_text=False)
        
        for rd in range(self.args.rounds-1):
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
                    prompt_template=prompts["debate_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=torch.cat(merged_other_embs, dim=0),                    
                )
                if agent.user_embs_fr is not None:
                    agent.history_embs = torch.cat([
                        agent.history_embs, 
                        agent.user_embs_fr, 
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr, 
                    ], dim=0)
                else:
                    agent.history_embs = torch.cat([
                        agent.history_embs,
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr, 
                    ], dim=0)
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["debate_prompt"].replace("{all_other_response}", all_other_resp)
                })
                if rd == self.args.rounds-2:
                    agent.generate(agent.history_embs, generate_text=True)
                else:
                    agent.generate(agent.history_embs, generate_text=False)
        for agent in self.agents:
            agent.final_output_text = agent.assistant_output_text[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time       
    
    


class LatentStage2DebateTask(DebateTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt

        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_embs)
        
        for rd in range(self.args.rounds-1):
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
                    prompt_template=prompts["debate_prompt"],
                    placeholder="{all_other_response}",
                    input_embs=torch.cat(merged_other_embs, dim=0),                    
                )
                if agent.user_embs_fr is not None:
                    agent.history_embs = torch.cat([
                        agent.history_embs, 
                        agent.user_embs_fr, 
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr, 
                    ], dim=0)
                else:
                    agent.history_embs = torch.cat([
                        agent.history_embs,
                        user_embs, 
                        agent.user_embs_ed, 
                        agent.assistant_embs_fr, 
                    ], dim=0)
                agent.history_msgs.append({
                    "role": "user",
                    "content": prompts["debate_prompt"].replace("{all_other_response}", all_other_resp)
                })
                agent.generate(agent.history_embs)
        for agent in self.agents:
            agent.final_output_text = agent.assistant_output_text[-1]
        end_time = time.perf_counter()
        for agent in self.agents:
            agent.history = agent.history_msgs
        return end_time - start_time                 
    
    
class HybridDebateTask(DebateTask):
    """
    Debate task supporting mixed LatentStage2Agent and NlAgent.
    
    Key design:
    - LatentStage2Agent reads other agents' outputs as embeddings.
      If the other agent is NlAgent, convert its text output to embedding first.
    - NlAgent reads other agents' outputs as plain text.
      If the other agent is LatentStage2Agent, use its assistant_output_text.
    """
 
    def _get_output_text(self, agent, round_idx: int) -> str:
        """Get plain text output of any agent at a given round."""
        return agent.assistant_output_text[round_idx] if agent.agent_type == 'latent2' \
            else agent.assistant_output[round_idx]
 
    def _get_output_embs(self, agent, round_idx: int,
                         receiver: "LatentStage2Agent") -> torch.Tensor:
        """
        Get embedding output of any agent at a given round,
        as seen by a LatentStage2Agent receiver.
        
        - If sender is LatentStage2Agent: use its stored output_embed directly.
        - If sender is NlAgent: convert its text output to embedding via receiver.
        """
        if agent.agent_type == 'latent2':
            return agent.assistant_output[round_idx]
        else:
            # NlAgent only has text; convert to embedding using receiver's tokenizer
            text = agent.assistant_output[round_idx]
            return receiver.text_to_embedding(text)
 
    def run_func(self, prompts):
        import time
        start_time = time.perf_counter()
        agent_cnt = self.args.agent_cnt
 
        # ── Round 0: each agent generates independently ──
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            if agent.agent_type == 'latent2':
                agent.generate(agent.history_embs)
            else:
                agent.generate(agent.history_msgs)
 
        # ── Subsequent rounds ──
        for rd in range(self.args.rounds - 1):
            for cur in range(agent_cnt):
                agent = self.agents[cur]
 
                if agent.agent_type == 'latent2':
                    # ── HyLaT agent: build embedding context from all others ──
                    merged_other_embs = []
                    all_other_resp_text = ""
 
                    for other in range(agent_cnt):
                        if cur == other:
                            continue
                        other_agent = self.agents[other]
 
                        # get embedding (converting NL text if necessary)
                        other_embs = self._get_output_embs(other_agent, rd, agent)
                        other_response_embs = get_merged_embedding_for_latent(
                            t2e_func=agent.text_to_embedding,
                            prompt_template=prompts["other_response_prompt"],
                            placeholder="{other_response}",
                            input_embs=other_embs,
                        )
                        merged_other_embs.append(other_response_embs)
 
                        # also build text version for history_msgs
                        other_text = self._get_output_text(other_agent, rd)
                        all_other_resp_text += prompts["other_response_prompt"].replace(
                            "{other_response}", other_text
                        )
 
                    user_embs = get_merged_embedding_for_latent(
                        t2e_func=agent.text_to_embedding,
                        prompt_template=prompts["debate_prompt"],
                        placeholder="{all_other_response}",
                        input_embs=torch.cat(merged_other_embs, dim=0),
                    )
 
                    if agent.user_embs_fr is not None:
                        agent.history_embs = torch.cat([
                            agent.history_embs,
                            agent.user_embs_fr,
                            user_embs,
                            agent.user_embs_ed,
                            agent.assistant_embs_fr,
                        ], dim=0)
                    else:
                        agent.history_embs = torch.cat([
                            agent.history_embs,
                            user_embs,
                            agent.user_embs_ed,
                            agent.assistant_embs_fr,
                        ], dim=0)
 
                    agent.history_msgs.append({
                        "role": "user",
                        "content": prompts["debate_prompt"].replace(
                            "{all_other_response}", all_other_resp_text
                        ),
                    })
                    agent.generate(agent.history_embs)
 
                else:
                    # ── NL agent: build plain text context from all others ──
                    all_other_resp_text = ""
                    for other in range(agent_cnt):
                        if cur == other:
                            continue
                        other_text = self._get_output_text(self.agents[other], rd)
                        all_other_resp_text += prompts["other_response_prompt"].replace(
                            "{other_response}", other_text
                        )
 
                    agent.history_msgs.append({
                        "role": "user",
                        "content": prompts["debate_prompt"].replace(
                            "{all_other_response}", all_other_resp_text
                        ),
                    })
                    agent.generate(agent.history_msgs)
 
        # ── Collect final outputs ──
        for agent in self.agents:
            agent.final_output_text = self._get_output_text(agent, -1)
            agent.history = agent.history_msgs
 
        end_time = time.perf_counter()
        return end_time - start_time