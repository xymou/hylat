import os
import gym
import json
import yaml
import copy
import time
import torch
import requests
from typing import Union, List

from src.tasks.utils import (
    get_merged_embedding_for_cipher,
    get_merged_ids_mask_hs_for_sde,
)
from src.tasks.base_task import BaseTask

import src.react.wikienv as wikienv
import src.react.wrappers as wrappers


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1


class WorkflowTask(BaseTask):
    def __init__(self, agents, dataset, args):
        super().__init__(agents, dataset, args)
        self.rounds = len(agents)
        if len(self.agents) > 0:
            self.agents = [self.agents[0]]
            self.args.agent_cnt = 1
        with open(args.prompt_file, "r") as fin:
            self.prompt_template = json.load(fin)[args.dataset]

        env = wikienv.WikiEnv()
        if args.dataset == "hotpotqa":
            env = wrappers.HotPotQAWrapper(env, split="dev")
        elif args.dataset == "strategyqa":
            env = wrappers.StrategyQAWrapper(env, split="dev")
        elif args.dataset == "fever":
            env = wrappers.FeverWrapper(env, split="dev")
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        env = wrappers.LoggingWrapper(env)
        self.env = env

    def run(self, data):
        question_prompt = self.env.reset(idx = data["test_id"])
        question = data["question"]
        ground_truth = data["answer"]
        detail = {
            "question": question, 
            "ground_truth": ground_truth,
        }
        prompts = copy.deepcopy(self.prompt_template)
        prompts["direct"] = prompts["direct"].replace("{question}", question)
        prompts["workflow"] += question_prompt
        ret, info, run_time, action_list = self.run_func(prompts)

        detail = {
            "question": question, 
            "ground_truth": ground_truth,
            "action_list": action_list,
            "result": ret,
            "info": info,
            "run_time(s)": run_time,
        }
        eval_ret = self.dataset.evaluate(output=info["answer"], test_id=data["test_id"])
        detail["predict"] = eval_ret["predict"]
        detail["evaluate_predict"] = eval_ret["evaluate_predict"]
        for met in self.dataset.eval_metrics:
            try:
                detail[met] = eval_ret[met].item()
            except:
                detail[met] = eval_ret[met]
        return detail
    
    def run_func(self, prompts):
        raise NotImplementedError

    def generate_result(self, details):
        res = {
            "run_time(s)": sum([d["run_time(s)"] for d in details]) / len(details),
        }
        ave = {}
        for met in self.dataset.eval_metrics:
            ave[met] = sum([d[met] for d in details]) / len(details)
        res["average"] = ave
        return res


class SingleWorkflowTask(WorkflowTask):
    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent = self.agents[0]
        prompt = prompts["direct"]
        agent.init_history(first_user_prompt=prompt)
        output = agent.generate(agent.history_msgs)
        agent.history = agent.history_msgs
        end_time = time.perf_counter()
        info = dict(answer=output)
        return None, info, end_time - start_time, None


class NlWorkflowTask(WorkflowTask):
    def llm_generate(self, agent, prompt, add_assistant_prompt, stop):
        agent.init_history(first_user_prompt=prompt)
        text = agent.generate(
            messages=agent.history_msgs, 
            add_assistant_prompt=add_assistant_prompt
        )
        for stop_word in stop:
            if stop_word in text:
                text = text.split(stop_word)[0]
        return text

    def run_func(self, prompts: dict):
        prompt = prompts["workflow"]
        start_time = time.perf_counter()
        n_calls, n_badcalls = 0, 0
        agent = self.agents[0]
        action_list = []
        for i in range(1, self.rounds+1):
            n_calls += 1
            text = self.llm_generate(agent, prompt, f"Thought {i}:", stop=[f"\nObservation {i}:"])
            flag = False
            try:
                thought, action = text.strip().split(f"\nAction {i}: ")
            except:
                flag = True
            if flag or len(action) == 0:
                # print('ohh...', text)
                n_badcalls += 1
                n_calls += 1
                thought = text.strip().split('\n')[0]
                action = self.llm_generate(agent, prompt, f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
            if len(action) == 0:
                action = "None"
            obs, r, done, info = step(self.env, action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')
            action_list.append({
                "idx": i,
                "thought": thought,
                "action": action, 
                "observation": obs,
            })
            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            prompt += step_str
            if done:
                break
        if not done:
            obs, r, done, info = step(self.env, "finish[]")
        info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
        end_time = time.perf_counter()
        return r, info, end_time - start_time, action_list


class CipherWorkflowTask(WorkflowTask):
    def __init__(self, agents, dataset, args):
        super().__init__(agents, dataset, args)
        if len(agents) > 0:
            agent = agents[0]
            agent.init_history(first_user_prompt = "")
            len_user_fr = agent.user_embs_fr.shape[0]
            self.system_embs_fr = agent.history_embs[:len_user_fr, :]

    def truncate_front(self, embeds, stop_word):
        tokenizer = self.agents[0].tokenizer
        ids = self.agents[0].get_token_ids(embeds)
        origin_text = tokenizer.decode(ids, skip_special_tokens=True)
        if stop_word not in origin_text:
            return embeds
        aim_text = origin_text.strip().split(stop_word)[0]
        aim_ids = tokenizer.encode(aim_text, add_special_tokens=False)
        trunc_pos = len(aim_ids)
        # 将 len*dim 的 embeds 截断为 trunc_pos*dim
        output_embeds = embeds[:trunc_pos, :].clone()
        return output_embeds

    def truncate_end(self, embeds, stop_word):
        tokenizer = self.agents[0].tokenizer
        ids = self.agents[0].get_token_ids(embeds)
        origin_text = tokenizer.decode(ids, skip_special_tokens=True)
        if stop_word not in origin_text:
            return embeds
        front_text = origin_text.strip().split(stop_word)[0] + stop_word
        front_ids = tokenizer.encode(front_text, add_special_tokens=False)
        trunc_pos = len(front_ids)
        output_embeds = embeds[trunc_pos:, :].clone()
        return output_embeds

    def llm_generate(self, agent, input_embeds, stop):
        output_embeds = agent.generate(input_embeds)
        for stop_word in stop:
            output_embeds = self.truncate_front(output_embeds, stop_word)
        return output_embeds

    def run_func(self, prompts: dict):
        prompt = prompts["workflow"]
        start_time = time.perf_counter()
        n_calls, n_badcalls = 0, 0
        agent = self.agents[0]
        action_list = []
        for i in range(1, self.rounds+1):
            n_calls += 1
            if i == 1:
                agent.init_history(first_user_prompt=prompt)
                step_embeds = agent.history_embs
            
            input_embeds = torch.cat([
                step_embeds, 
                agent.text_to_embedding(f"Thought {i}: "),
            ], dim=0)
            output_embeds = self.llm_generate(
                agent, 
                input_embeds=input_embeds, 
                stop=[f"\nObservation"],
            )
            output_text = agent.get_human_output(output_embeds)

            if f"\nAction {i}:" in output_text:
                flag = True
                thought_embeds = self.truncate_front(output_embeds, stop_word=f"\nAction {i}:")
                action_embeds = self.truncate_end(output_embeds, stop_word=f"\nAction {i}:")
                # del output_embeds
            if f"\nAction {i}:" not in output_text or action_embeds.shape[0] == 0:
                n_badcalls += 1
                n_calls += 1
                thought_embeds = self.truncate_front(output_embeds, stop_word="\n")
                assis_embeds = get_merged_embedding_for_cipher(
                    t2e_func=agent.text_to_embedding,
                    prompt_template=f"Thought {i}: {{placeholder}}\nAction {i}:",
                    placeholder="{placeholder}",
                    input_embs=thought_embeds,
                )
                action_embeds = self.llm_generate(
                    agent=agent,
                    input_embeds=torch.cat([step_embeds, assis_embeds], dim=0),
                    stop=[f"\n"],
                )
            thought_text = agent.get_human_output(thought_embeds)
            action_text = agent.get_human_output(action_embeds).strip()
            if action_text.endswith("<|im_end|>"):
                action_text = action_text[:-len("<|im_end|>")]
            if len(action_text) == 0:
                obs, r, done, info = step(self.env, "None")
            else:
                obs, r, done, info = step(self.env, action_text[0].lower() + action_text[1:])
            obs = obs.replace('\\n', '')
            action_list.append({
                "idx": i,
                "thought": thought_text,
                "action": action_text, 
                "observation": obs,
            })
            if done:
                break
            new_prompt = prompt + f"Thought {i}: {thought_text}\nAction {i}: {action_text}\nObservation {i}: {obs}\n"
            thought_embeds = get_merged_embedding_for_cipher(
                t2e_func=agent.text_to_embedding,
                prompt_template=f"Thought {i}: {{thought}}\n",
                placeholder="{thought}",
                input_embs=thought_embeds,
            )
            action_embeds = get_merged_embedding_for_cipher(
                t2e_func=agent.text_to_embedding,
                prompt_template=f"Action {i}: {{action}}\n",
                placeholder="{action}",
                input_embs=action_embeds,
            )
            step_embeds = get_merged_embedding_for_cipher(
                t2e_func=agent.text_to_embedding,
                prompt_template=f"{prompt}{{placeholder}}\nObservation {i}: {obs}\n",
                placeholder="{placeholder}",
                input_embs=torch.cat([thought_embeds, action_embeds], dim=0),
            )
            step_embeds = torch.cat([
                self.system_embs_fr, 
                step_embeds, 
                agent.assistant_embs_fr,
            ])
            prompt = new_prompt
        if not done:
            obs, r, done, info = step(self.env, "finish[]")
        info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
        end_time = time.perf_counter()
        return r, info, end_time - start_time, action_list




class SDEWorkflowTask(WorkflowTask):
    def __init__(self, agents, dataset, args):
        super().__init__(agents, dataset, args)
        self.edit_layer_idx = args.edit_layer_idx
        if len(agents) > 0:
            agent = agents[0]
            agent.init_history(first_user_prompt = "")
            self.system_prompt_fr_ids = agent.history_ids[:-len(agent.user_prompt_ed + agent.assistant_prompt_fr)]
            self.system_prompt_fr_text = agent.tokenizer.decode(self.system_prompt_fr_ids, skip_special_tokens=False)

    def truncate_front(self, ids, hss, stop_word):
        tokenizer = self.agents[0].tokenizer
        origin_text = tokenizer.decode(ids, skip_special_tokens=True)
        if stop_word not in origin_text:
            return ids, hss
        aim_text = origin_text.strip().split(stop_word)[0]
        aim_ids = tokenizer.encode(aim_text, add_special_tokens=False)
        trunc_pos = len(aim_ids)
        if trunc_pos > len(ids):
            trunc_pos = len(ids)
            aim_ids = aim_ids[:trunc_pos]
        output_hs = {}
        for layer_idx, hs in hss.items():
            output_hs[layer_idx] = hs[:, :trunc_pos, :].clone()
        return aim_ids, output_hs

    def truncate_end(self, ids, hss, stop_word):
        tokenizer = self.agents[0].tokenizer
        origin_text = tokenizer.decode(ids, skip_special_tokens=True)
        if stop_word not in origin_text:
            return ids, hss
        front_text = origin_text.strip().split(stop_word)[0] + stop_word
        front_ids = tokenizer.encode(front_text, add_special_tokens=False)
        trunc_pos = len(front_ids)
        output_hs = {}
        for layer_idx, hs in hss.items():
            output_hs[layer_idx] = hs[:, trunc_pos:, :].clone()
        return ids[trunc_pos:], output_hs
    
    def llm_generate(self, **kwargs):
        # 从 kwargs 里去掉 stop 这一项
        stop = kwargs.pop("stop", None)
        output_ids, output_hs = self.agents[0].generate(**kwargs)
        for stop_word in stop:
            output_ids, output_hs = self.truncate_front(output_ids, output_hs, stop_word)
        return output_ids, output_hs

    def run_func(self, prompts: dict):
        prompt = prompts["workflow"]
        start_time = time.perf_counter()
        n_calls, n_badcalls = 0, 0
        agent = self.agents[0]
        action_list = []
        assistant_fr = agent.tokenizer.decode(agent.user_prompt_ed + agent.assistant_prompt_fr, skip_special_tokens=False)
        for i in range(1, self.rounds+1):
            n_calls += 1
            if i == 1:
                agent.init_history(first_user_prompt = prompt)
                input_ids = agent.history_ids + agent.tokenizer.encode("Thought 1: ", add_special_tokens=False)
                output_ids, output_hs = self.llm_generate(
                    input_ids=input_ids,
                    if_edit=False, 
                    edit_layer_idx=self.edit_layer_idx,
                    stop=[f"\nObservation"],
                )
            else:
                output_ids, output_hs = self.llm_generate(
                    input_ids=step_ids, 
                    if_edit=True,
                    edit_layer_idx=self.edit_layer_idx,
                    edit_mask=step_mask,
                    edit_tensor=step_hs,
                    stop=[f"\nObservation"],
                )
            output_text = agent.tokenizer.decode(output_ids, skip_special_tokens=False)
            if f"\nAction {i}:" in output_text:
                flag = True
                thought_ids, thought_hs = self.truncate_front(output_ids, output_hs, stop_word=f"\nAction {i}:")
                action_ids, action_hs = self.truncate_end(output_ids, output_hs, stop_word=f"\nAction {i}:")
                # del output_ids, output_hs
            if f"\nAction {i}:" not in output_text or len(action_ids) == 0:
                n_badcalls += 1
                n_calls += 1
                thought_ids, thought_hs = self.truncate_front(output_ids, output_hs, stop_word="\n")
                # print(len(thought_ids), thought_hs[self.edit_layer_idx[0]].shape)
                input_ids, input_mask, input_hs = get_merged_ids_mask_hs_for_sde(
                    tokenizer=agent.tokenizer, 
                    prompt_template=f"{self.system_prompt_fr_text}{prompt}{assistant_fr}Thought {i}: {{placeholder}}\nAction {i}:",
                    placeholder="{placeholder}",
                    input_ids=thought_ids, 
                    input_mask=torch.zeros(len(thought_ids), dtype=torch.bool),
                    input_hs=thought_hs,
                )
                # print(len(input_ids), input_mask.shape, input_hs[self.edit_layer_idx[0]].shape)
                action_ids, action_hs = self.llm_generate(
                    input_ids=input_ids,
                    if_edit=True,
                    edit_layer_idx=self.edit_layer_idx,
                    edit_mask=input_mask,
                    edit_tensor=input_hs,
                    stop=[f"\n"],
                )
            action_ids, action_hs = self.truncate_front(action_ids, action_hs, stop_word="\n")
            action_text = agent.tokenizer.decode(action_ids, skip_special_tokens=False).strip()
            if len(action_text) == 0:
                obs, r, done, info = step(self.env, "None")
            else:
                obs, r, done, info = step(self.env, action_text[0].lower() + action_text[1:])
            obs = obs.replace('\\n', '')
            action_list.append({
                "idx": i,
                "thought": agent.tokenizer.decode(thought_ids, skip_special_tokens=False),
                "action": action_text, 
                "observation": obs,
            })
            if done:
                break
            thought_text = agent.tokenizer.decode(thought_ids, skip_special_tokens=False)
            new_prompt = prompt + f"Thought {i}: {thought_text}\nAction {i}: {action_text}\nObservation {i}: {obs}\n" 
            thought_ids, thought_mask, thought_hs = get_merged_ids_mask_hs_for_sde(
                tokenizer=agent.tokenizer, 
                prompt_template=f"Thought {i}: {{thought}}\n",
                placeholder="{thought}",
                input_ids=thought_ids, 
                input_mask=torch.zeros(len(thought_ids), dtype=torch.bool),
                input_hs=thought_hs,
            )
            action_ids, action_mask, action_hs = get_merged_ids_mask_hs_for_sde(
                tokenizer=agent.tokenizer, 
                prompt_template=f"Action {i}: {{action}}\n",
                placeholder="{action}",
                input_ids=action_ids, 
                input_mask=torch.zeros(len(action_ids), dtype=torch.bool),
                input_hs=action_hs,
            )
            step_ids, step_mask, step_hs = get_merged_ids_mask_hs_for_sde(
                tokenizer=agent.tokenizer,
                prompt_template=f"{self.system_prompt_fr_text}{prompt}{{placeholder}}\nObservation {i}: {obs}\n{assistant_fr}Thought {i+1}: ",
                placeholder="{placeholder}",
                input_ids=thought_ids+action_ids,
                input_mask=torch.cat([thought_mask, action_mask]),
                input_hs={
                    layer_idx: torch.cat([thought_hs[layer_idx], action_hs[layer_idx]], dim=1)
                    for layer_idx in thought_hs
                }
            )
            prompt = new_prompt
        if not done:
            obs, r, done, info = step(self.env, "finish[]")
        info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
        end_time = time.perf_counter()
        return r, info, end_time - start_time, action_list
