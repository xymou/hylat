import re
import copy
import time
import yaml
from src.tasks.base_task import BaseTask
import torch
from src.tasks.utils import (
    get_merged_embedding_for_cipher,
    get_merged_ids_mask_hs_for_sde,
    get_merged_embedding_for_latent
)

class TrustGameTask(BaseTask):
    """
    信任博弈实验 Task
    支持6种博弈变体，通过 args.game_type 指定：
      trust / dictator / map_trust / risky_dictator / lottery / repeated
    支持两种运行模式，通过 args.method 指定：
      single  — 只用 agents[0] 同时扮演 trustor 和 trustee
      multi   — agents[0] 扮演 trustor，agents[1] 扮演 trustee
    """

    VALID_GAME_TYPES = {
        "trust", "dictator", "map_trust",
        "risky_dictator", "lottery", "repeated"
    }

    def __init__(self, agents, dataset, args):
        super().__init__(agents, dataset, args)

        assert hasattr(args, "game_type") and args.game_type in self.VALID_GAME_TYPES, \
            f"args.game_type must be one of {self.VALID_GAME_TYPES}"

        if args.method == "single":
            self.agents = [self.agents[0]]
            self.args.agent_cnt = 1
        elif args.method == "multi":
            assert len(self.agents) >= 2, "multi mode needs at least 2 agents"
            self.agents = self.agents[:2]
            self.args.agent_cnt = 2

        with open(args.prompt_file) as fin:
            self.prompt_template = yaml.safe_load(fin)["trust_game"][args.game_type]

    # ------------------------------------------------------------------
    # 核心入口
    # ------------------------------------------------------------------
    def run(self, data):
        """
        data 字段：
          - persona_trustor: dict  (name, age, gender, job, background)
          - persona_trustee: dict  (可选，single mode 下忽略)
          - prob_p: float          (MAP/risky/lottery 博弈专用，范围 0~1)
          - initial_amount: int    (默认 10)
          - test_id: str
        """
        persona_trustor = data["persona_trustor"]
        persona_trustee = data.get("persona_trustee", persona_trustor)
        initial_amount   = data.get("initial_amount", 10)
        prob_p           = data.get("prob_p", None)   # MAP/risky 博弈需要

        detail = {
            "persona_trustor": persona_trustor,
            "persona_trustee": persona_trustee,
            "game_type":       self.args.game_type,
            "initial_amount":  initial_amount,
        }
        if prob_p is not None:
            detail["prob_p"] = prob_p

        # 渲染 prompt 模板
        prompts = self._render_prompts(
            persona_trustor, persona_trustee, initial_amount, prob_p
        )

        # 运行对应 mode
        run_time = self.run_func(prompts, initial_amount)

        # 解析结果
        trustor_agent = self.agents[0]
        trustee_agent = self.agents[-1]   # single mode 下和 trustor 相同

        sent   = self._parse_amount(trustor_agent.final_output_text, key="give")
        returned = self._parse_amount(trustee_agent.final_output_text, key="return") \
                   if self.args.game_type not in ("dictator",) else 0

        # 有效性检验（VRR）
        valid_sent     = self._is_valid(sent,     0, initial_amount)
        valid_returned = self._is_valid(returned, 0, sent * 3) \
                         if sent is not None else False

        detail.update({
            "trustor_output":    trustor_agent.final_output_text,
            "trustee_output":    trustee_agent.final_output_text,
            "trustor_history":   trustor_agent.history,
            "trustee_history":   trustee_agent.history,
            "amount_sent":       sent,
            "amount_returned":   returned,
            "valid_sent":        valid_sent,
            "valid_returned":    valid_returned,
            # 核心指标：信任率（是否发出正数）和互惠率
            "trust_score":       (sent / initial_amount) if valid_sent else None,
            "reciprocate_score": (returned / (sent * 3)) if (valid_sent and sent > 0) else None,
            "run_time(s)":       run_time,
        })

        # MAP / risky 博弈额外记录 trust/no-trust 决策
        if self.args.game_type in ("map_trust", "risky_dictator", "lottery"):
            detail["trust_decision"] = self._parse_binary_decision(
                trustor_agent.final_output_text
            )
        return detail

    # ------------------------------------------------------------------
    # run_func 由子类实现
    # ------------------------------------------------------------------
    def run_func(self, prompts, initial_amount):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 汇总统计
    # ------------------------------------------------------------------
    def generate_result(self, details):
        valid_details = [d for d in details if d["valid_sent"]]
        n_total = len(details)
        n_valid = len(valid_details)

        res = {
            "game_type":       self.args.game_type,
            "n_total":         n_total,
            "n_valid":         n_valid,
            "VRR(%)":          round(n_valid / n_total * 100, 2) if n_total else 0,
            "avg_sent":        self._safe_mean([d["amount_sent"]    for d in valid_details]),
            "avg_returned":    self._safe_mean([d["amount_returned"] for d in valid_details]),
            "avg_trust_score": self._safe_mean([d["trust_score"]    for d in valid_details]),
            "avg_run_time(s)": self._safe_mean([d["run_time(s)"]    for d in details]),
        }

        # MAP / risky / lottery 博弈额外统计信任率曲线所需数据
        if self.args.game_type in ("map_trust", "risky_dictator", "lottery"):
            trust_decisions = [d["trust_decision"] for d in details if d.get("trust_decision") is not None]
            res["trust_rate(%)"] = round(
                sum(trust_decisions) / len(trust_decisions) * 100, 2
            ) if trust_decisions else None

        return res

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _render_prompts(self, persona_t, persona_e, initial_amount):
        prompts = copy.deepcopy(self.prompt_template)
        # background 已经是完整 persona 描述，直接替换
        prompts["role_prompt"] = prompts["role_prompt"].replace(
            "{background_trustor}", persona_t["background"]
        )
        # trustor_prompt / feedback_prompt / trustee_prompt 里只有金额占位符
        for key in ("trustor_prompt", "feedback_prompt",
                    "trustee_prompt", "trustee_feedback_prompt"):
            if key in prompts:
                prompts[key] = prompts[key].replace(
                    "{initial_amount}", str(initial_amount)
                )
        return prompts

    @staticmethod
    def _parse_amount(text, key="give"):
        if not text:
            return None
        
        patterns = [
            # 论文标准格式
            rf"I will {key}\s+\$?(\d+(?:\.\d+)?)",
            rf"Finally,\s+I will {key}\s+\$?(\d+(?:\.\d+)?)",
            # "I will give back / return back" 等变体
            rf"I will {key}\s+back\s+\$?(\d+(?:\.\d+)?)",
            # "I choose to give / I decide to give"
            rf"I (?:choose|decide) to {key}\s+\$?(\d+(?:\.\d+)?)",
            # "The answer is X"
            r"[Tt]he answer is\s+\$?(\d+(?:\.\d+)?)",
            r"[Tt]he answer is:\s+\$?(\d+(?:\.\d+)?)",
            # "My answer is X"
            r"[Mm]y answer is\s+\$?(\d+(?:\.\d+)?)",
            # "I would give / return X"
            rf"I would {key}\s+\$?(\d+(?:\.\d+)?)",
            # "give/return $X" 或 "give/return X dollars"
            rf"{key}\s+\$(\d+(?:\.\d+)?)",
            rf"{key}\s+(\d+(?:\.\d+)?)\s+dollars?",
            # "my decision is to give X"
            rf"my decision is to {key}\s+\$?(\d+(?:\.\d+)?)",
            # 兜底：独立的 "X dollars" （放最后避免误匹配）
            r"(\d+(?:\.\d+)?)\s+dollars?",
        ]
        
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return float(m.group(1))
        
        # 最后兜底：找文本末尾的独立数字（模型只输出一个数的情况）
        m = re.search(r"\b(\d+(?:\.\d+)?)\b(?!.*\b\d+(?:\.\d+)?\b)", text.strip())
        if m:
            return float(m.group(1))
        
        return None

    @staticmethod
    def _parse_binary_decision(text):
        """MAP / risky / lottery 博弈：解析 trust / no-trust 二元决策"""
        text_lower = text.lower()
        if re.search(r"\bi (will |choose to |decide to )?trust\b", text_lower):
            return 1
        if re.search(r"\bi (will |choose |decide )?(not|refuse) (to )?trust\b", text_lower):
            return 0
        return None

    @staticmethod
    def _is_valid(amount, lo, hi):
        return amount is not None and lo <= amount <= hi

    @staticmethod
    def _safe_mean(values):
        valid = [v for v in values if v is not None]
        return round(sum(valid) / len(valid), 4) if valid else None


# ======================================================================
# 子类：single agent（trustor 和 trustee 是同一个 agent，分两次调用）
# ======================================================================
class SingleTrustGameTask(TrustGameTask):
    def run_func(self, prompts, initial_amount):
        start = time.perf_counter()
        agent = self.agents[0]

        # --- Phase 1: trustor 决策 ---
        agent.init_history(first_user_prompt=prompts["trustor_prompt"])
        trustor_output = agent.generate(agent.history_msgs)
        agent.final_output_text = trustor_output  # trustor 结果暂存

        # 解析发出金额，用于构造 trustee prompt
        sent = TrustGameTask._parse_amount(trustor_output, key="give") or 0

        # --- Phase 2: trustee 决策（复用同一 agent，重置历史）---
        trustee_prompt = prompts["trustee_prompt"] \
            .replace("{amount_sent}",     str(int(sent))) \
            .replace("{amount_received}", str(int(sent * 3)))

        # 保存 trustor history，再重置给 trustee 用
        trustor_history = agent.history_msgs[:]
        agent.init_history(first_user_prompt=trustee_prompt)
        trustee_output = agent.generate(agent.history_msgs)

        # 把 trustee 的结果写回（generate_result 只看 final_output_text）
        # 这里用 history 区分两个阶段
        agent.history = {
            "trustor": trustor_history,
            "trustee": agent.history_msgs[:],
        }
        # final_output_text 存 trustee 结果（trustor 结果在 detail 里单独记）
        agent.final_output_text = trustee_output

        return time.perf_counter() - start


# ======================================================================
# 子类：multi agent（agents[0]=trustor, agents[1]=trustee）
# ======================================================================
class RepeatedTrustGameTask(BaseTask):
    def __init__(self, agents, dataset, args):
        super().__init__(agents, dataset, args)

        if args.method == "single":
            self.agents = [self.agents[0]]
            self.args.agent_cnt = 1
        else:
            self.agents = self.agents[:2]
            self.args.agent_cnt = 2

        with open(args.prompt_file) as fin:
            self.prompt_template = yaml.safe_load(fin)["repeated_trust_game"]

    def run(self, data):
        persona_trustor = data["persona_trustor"]
        persona_trustee = data.get("persona_trustee", persona_trustor)
        initial_amount  = data.get("initial_amount", 10)

        prompts = self._render_prompts(persona_trustor, persona_trustee, initial_amount)
        run_time = self.run_func(prompts, initial_amount)

        trustor_agent = self.agents[0]
        trustee_agent = self.agents[-1]

        return {
            "persona_trustor":  persona_trustor,
            "initial_amount":   initial_amount,
            "all_sent":         trustor_agent.all_sent,
            "all_returned":     trustee_agent.all_returned,
            "trustor_history":  trustor_agent.history,
            "trustee_history":  trustee_agent.history,
            "valid_rounds":     trustor_agent.valid_rounds,
            "run_time(s)":      run_time,
        }

    def run_func(self, prompts, initial_amount):
        raise NotImplementedError

    def generate_result(self, details):
        n = len(details)
        rounds = getattr(self.args, "rounds", 5)
        avg_sent_by_round = [
            self._safe_mean([d["all_sent"][rd] for d in details if len(d["all_sent"]) > rd])
            for rd in range(rounds)
        ]
        return {
            "n_total":              n,
            "rounds":               rounds,
            "avg_sent_by_round":    avg_sent_by_round,
            "avg_sent_overall":     self._safe_mean([s for d in details for s in d["all_sent"]]),
            "avg_returned_overall": self._safe_mean([r for d in details for r in d["all_returned"]]),
            "avg_run_time(s)":      self._safe_mean([d["run_time(s)"] for d in details]),
        }

    # ------------------------------------------------------------------ 
    # 工具方法
    # ------------------------------------------------------------------
    def _render_prompts(self, persona_t, persona_e, initial_amount):
        prompts = copy.deepcopy(self.prompt_template)
        # background 已经是完整 persona 描述，直接替换
        prompts["role_prompt"] = prompts["role_prompt"].replace(
            "{background_trustor}", persona_t
        )
        # trustor_prompt / feedback_prompt / trustee_prompt 里只有金额占位符
        for key in ("trustor_prompt", "feedback_prompt",
                    "trustee_prompt", "trustee_feedback_prompt"):
            if key in prompts:
                prompts[key] = prompts[key].replace(
                    "{initial_amount}", str(initial_amount)
                )
        return prompts

    def _fill_round_prompt(self, template, sent, returned=None, rd=None):
        """把当轮数值填入 prompt 字符串"""
        s = template \
            .replace("{amount_sent}",     str(int(sent))) \
            .replace("{amount_received}", str(int(sent * 3)))
        if returned is not None:
            s = s.replace("{amount_returned}", str(int(returned)))
        if rd is not None:
            s = s.replace("{round}", str(rd + 1))
        return s

    # ---------- agent 类型判断 ----------
    @staticmethod
    def _agent_type(agent):
        """统一判断 agent 类型，避免到处写 hasattr"""
        if hasattr(agent, "agent_type"):        # LatentStage2 / HyLaT 等已标注
            return agent.agent_type
        if hasattr(agent, "history_ids"):       # SDE
            return "sde"
        if hasattr(agent, "history_embs"):      # Cipher
            return "cipher"
        return "nl"

    # ---------- 各类型初始化 ----------
    def _init_agent(self, agent, prompt_text):
        t = self._agent_type(agent)
        agent.init_history(first_user_prompt=prompt_text)
        if t == "sde":
            agent.generate(
                input_ids=agent.history_ids,
                if_edit=False,
                edit_layer_idx=self.args.edit_layer_idx,
            )
        elif t in ("cipher", "latent2"):
            agent.generate(agent.history_embs)
        else:
            agent.generate(agent.history_msgs)

    # ---------- 追加一轮 user 消息并 generate ----------
    def _append_and_generate(self, agent, prompt_text, if_edit=False,
                              edit_mask=None, edit_tensor=None):
        t = self._agent_type(agent)

        if t == "nl":
            agent.history_msgs.append({"role": "user", "content": prompt_text})
            agent.generate(agent.history_msgs)

        elif t == "cipher":
            # user_embs = get_merged_embedding_for_cipher(
            #     t2e_func=agent.text_to_embedding,
            #     prompt_template=prompt_text,
            #     placeholder="",        # 纯文本，无需替换占位符
            #     input_embs=agent.text_to_embedding(""),
            # )
            # cipher 直接把文本 embed 后拼到 history_embs
            user_embs = agent.text_to_embedding(prompt_text)
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

        elif t == "latent2":
            # user_embs = get_merged_embedding_for_latent(
            #     t2e_func=agent.text_to_embedding,
            #     prompt_template=prompt_text,
            #     placeholder="",
            #     input_embs=agent.text_to_embedding(""),
            # )
            user_embs = agent.text_to_embedding(prompt_text)
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
            agent.history_msgs.append({"role": "user", "content": prompt_text})
            agent.generate(agent.history_embs)

        elif t == "sde":
            # SDE 需要把文本转成 ids 后拼入 history_ids
            new_ids = agent.tokenizer.encode(prompt_text, add_special_tokens=False)
            history_mask = torch.cat([
                torch.zeros(len(agent.history_ids) + len(agent.user_prompt_fr), dtype=torch.bool),
                torch.zeros(len(new_ids), dtype=torch.bool),
                torch.zeros(len(agent.user_prompt_ed) + len(agent.assistant_prompt_fr), dtype=torch.bool),
            ], dim=0)
            agent.history_ids = (
                agent.history_ids
                + agent.user_prompt_fr
                + new_ids
                + agent.user_prompt_ed
                + agent.assistant_prompt_fr
            )
            agent.generate(
                input_ids=agent.history_ids,
                if_edit=if_edit,
                edit_layer_idx=self.args.edit_layer_idx,
                edit_mask=edit_mask if edit_mask is not None else history_mask,
                edit_tensor=edit_tensor if edit_tensor is not None else {},
            )

    # ---------- 获取最新一轮的文本输出 ----------
    @staticmethod
    def _get_latest_text(agent):
        t = RepeatedTrustGameTask._agent_type(agent)
        if t == "latent2":
            return agent.assistant_output_text[-1]
        elif t == "sde":
            return agent.get_human_output(agent.assistant_ids[-1])
        elif t == "cipher":
            return agent.get_human_output(agent.assistant_output[-1])
        else:
            return agent.assistant_output[-1]

    # ---------- 获取 history 用于记录 ----------
    @staticmethod
    def _get_history(agent):
        t = RepeatedTrustGameTask._agent_type(agent)
        if t in ("nl", "latent2"):
            return agent.history_msgs
        elif t in ("cipher", "latent2"):
            return agent.get_human_output(agent.history_embs)
        elif t == "sde":
            return agent.get_human_output(agent.history_ids)
        return agent.history_msgs

    @staticmethod
    def _parse_amount(text, key="give"):
        if not text:
            return None
        
        patterns = [
            # 论文标准格式
            rf"I will {key}\s+\$?(\d+(?:\.\d+)?)",
            rf"Finally,\s+I will {key}\s+\$?(\d+(?:\.\d+)?)",
            # "I will give back / return back" 等变体
            rf"I will {key}\s+back\s+\$?(\d+(?:\.\d+)?)",
            # "I choose to give / I decide to give"
            rf"I (?:choose|decide) to {key}\s+\$?(\d+(?:\.\d+)?)",
            # "The answer is X"
            r"[Tt]he answer is\s+\$?(\d+(?:\.\d+)?)",
            r"[Tt]he answer is:\s+\$?(\d+(?:\.\d+)?)",
            # "My answer is X"
            r"[Mm]y answer is\s+\$?(\d+(?:\.\d+)?)",
            # "I would give / return X"
            rf"I would {key}\s+\$?(\d+(?:\.\d+)?)",
            # "give/return $X" 或 "give/return X dollars"
            rf"{key}\s+\$(\d+(?:\.\d+)?)",
            rf"{key}\s+(\d+(?:\.\d+)?)\s+dollars?",
            # "my decision is to give X"
            rf"my decision is to {key}\s+\$?(\d+(?:\.\d+)?)",
            # 兜底：独立的 "X dollars" （放最后避免误匹配）
            r"(\d+(?:\.\d+)?)\s+dollars?",
        ]
        
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return float(m.group(1))
        
        # 最后兜底：找文本末尾的独立数字（模型只输出一个数的情况）
        m = re.search(r"\b(\d+(?:\.\d+)?)\b(?!.*\b\d+(?:\.\d+)?\b)", text.strip())
        if m:
            return float(m.group(1))
        
        return None

    @staticmethod
    def _is_valid(amount, lo, hi):
        return amount is not None and lo <= amount <= hi

    @staticmethod
    def _safe_mean(values):
        valid = [v for v in values if v is not None]
        return round(sum(valid) / len(valid), 4) if valid else None


# ======================================================================
# NL / Cipher / LatentStage2 子类（generate 接口一致，_append_and_generate 已统一处理）
# ======================================================================
class NlRepeatedTrustGameTask(RepeatedTrustGameTask):
    def run_func(self, prompts, initial_amount):
        start = time.perf_counter()
        trustor = self.agents[0]
        trustee = self.agents[-1]
        rounds  = getattr(self.args, "rounds", 5)

        self._init_agent(trustor, prompts["trustor_prompt"])
        all_sent, all_returned, valid_rounds = [], [], []

        for rd in range(rounds):
            # trustor 输出（第 0 轮已在 init 里 generate 过，直接取结果）
            trustor_text = self._get_latest_text(trustor)
            sent  = self._parse_amount(trustor_text, key="give")
            valid = self._is_valid(sent, 0, initial_amount)
            sent  = sent if valid else 0
            all_sent.append(sent)
            valid_rounds.append(valid)

            # trustee 决策
            trustee_prompt = self._fill_round_prompt(
                prompts["trustee_prompt"], sent, rd=rd
            )
            if rd == 0:
                self._init_agent(trustee, trustee_prompt)
            else:
                self._append_and_generate(trustee, trustee_prompt)
            returned = self._parse_amount(self._get_latest_text(trustee), key="return") or 0
            all_returned.append(returned)

            # 反馈给 trustor（最后一轮不需要）
            if rd < rounds - 1:
                feedback = self._fill_round_prompt(
                    prompts["feedback_prompt"], sent, returned=returned, rd=rd
                )
                self._append_and_generate(trustor, feedback)

        trustor.final_output_text = self._get_latest_text(trustor)
        trustee.final_output_text = self._get_latest_text(trustee)
        trustor.history     = self._get_history(trustor)
        trustee.history     = self._get_history(trustee)
        trustor.all_sent    = all_sent
        trustee.all_returned = all_returned
        trustor.valid_rounds = valid_rounds

        return time.perf_counter() - start


# Cipher / LatentStage2 复用完全相同的逻辑，_append_and_generate 内部已按类型分支
CipherRepeatedTrustGameTask   = NlRepeatedTrustGameTask
LatentRepeatedTrustGameTask   = NlRepeatedTrustGameTask


# ======================================================================
# SDE 子类（generate 签名不同，需要单独处理）
# ======================================================================
class SDERepeatedTrustGameTask(RepeatedTrustGameTask):
    def run_func(self, prompts, initial_amount):
        start = time.perf_counter()
        trustor = self.agents[0]
        trustee = self.agents[-1]
        rounds  = getattr(self.args, "rounds", 5)

        # SDE init：不带 edit
        trustor.init_history(first_user_prompt=prompts["trustor_prompt"])
        trustor.generate(
            input_ids=trustor.history_ids,
            if_edit=False,
            edit_layer_idx=self.args.edit_layer_idx,
        )

        all_sent, all_returned, valid_rounds = [], [], []

        for rd in range(rounds):
            trustor_text = trustor.get_human_output(trustor.assistant_ids[-1])
            sent  = self._parse_amount(trustor_text, key="give")
            valid = self._is_valid(sent, 0, initial_amount)
            sent  = sent if valid else 0
            all_sent.append(sent)
            valid_rounds.append(valid)

            # trustee
            trustee_prompt = self._fill_round_prompt(
                prompts["trustee_prompt"], sent, rd=rd
            )
            if rd == 0:
                trustee.init_history(first_user_prompt=trustee_prompt)
                trustee.generate(
                    input_ids=trustee.history_ids,
                    if_edit=False,
                    edit_layer_idx=self.args.edit_layer_idx,
                )
            else:
                self._append_and_generate(trustee, trustee_prompt)
            returned = self._parse_amount(
                trustee.get_human_output(trustee.assistant_ids[-1]), key="return"
            ) or 0
            all_returned.append(returned)

            # 反馈给 trustor
            if rd < rounds - 1:
                feedback = self._fill_round_prompt(
                    prompts["feedback_prompt"], sent, returned=returned, rd=rd
                )
                self._append_and_generate(trustor, feedback)

        trustor.final_output_text = trustor.get_human_output(trustor.assistant_ids[-1])
        trustee.final_output_text = trustee.get_human_output(trustee.assistant_ids[-1])
        trustor.history      = trustor.get_human_output(trustor.history_ids)
        trustee.history      = trustee.get_human_output(trustee.history_ids)
        trustor.all_sent     = all_sent
        trustee.all_returned = all_returned
        trustor.valid_rounds = valid_rounds

        return time.perf_counter() - start