import time
import torch
from src.tasks.debate import DebateTask
from src.tasks.utils import get_merged_embedding_for_latent


class HeteroLatentDebateTask(DebateTask):
    """
    Debate task for heterogeneous latent agents (e.g. Llama + Qwen).

    Both HeteroLatentAgent instances share the same CODI model but use
    different sub-models (codi1 / codi2). Their latent embeddings live in
    different hidden spaces (H1 vs H2), so before embedding concatenation
    the task projects the sender's output into the receiver's space via
    adapter_1to2 or adapter_2to1.  No <latent> placeholder mechanism is
    used at inference time.
    """

    def _project_for_receiver(
        self,
        embs: torch.Tensor,  # [seq_len, H_sender]
        sender_model_type: int,   # agent.agent_idx of the sender  (0=codi1, 1=codi2)
        receiver_model_type: int, # agent.agent_idx of the receiver
    ) -> torch.Tensor:
        """
        Project sender's embeddings into the receiver's hidden space.

        Uses agent.agent_idx (0/1 model type), NOT the position in self.agents,
        so that N-agent scenarios with repeated model types work correctly.
        Same-type agents share the same hidden space and need no projection.
        """
        if sender_model_type == receiver_model_type:
            return embs
        model = self.agents[0].model  # all agents share the same CODI instance
        adapter = model.adapter_1to2 if (sender_model_type == 0) else model.adapter_2to1
        with torch.no_grad():
            return adapter(
                embs.to(next(adapter.parameters()).dtype)
            ).to(embs.dtype)

    def run_func(self, prompts):
        start_time = time.perf_counter()
        agent_cnt  = self.args.agent_cnt

        # Round 0: each agent generates independently
        for agent in self.agents:
            agent.init_history(first_user_prompt=prompts["first_prompt"])
            agent.generate(agent.history_embs)

        # Subsequent rounds
        for rd in range(self.args.rounds - 1):
            for cur in range(agent_cnt):
                agent             = self.agents[cur]
                merged_other_embs = []
                all_other_resp    = ""

                for other in range(agent_cnt):
                    if cur == other:
                        continue
                    # Use agent_idx (model type 0/1) for projection, not list position
                    projected = self._project_for_receiver(
                        self.agents[other].assistant_output[rd],
                        sender_model_type=self.agents[other].agent_idx,
                        receiver_model_type=self.agents[cur].agent_idx,
                    )
                    other_response = get_merged_embedding_for_latent(
                        t2e_func=agent.text_to_embedding,
                        prompt_template=prompts["other_response_prompt"],
                        placeholder="{other_response}",
                        input_embs=projected,
                    )
                    merged_other_embs.append(other_response)
                    other_resp     = self.agents[other].assistant_output_text[rd]
                    all_other_resp += prompts["other_response_prompt"].replace(
                        "{other_response}", other_resp
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
                        "{all_other_response}", all_other_resp
                    ),
                })
                agent.generate(agent.history_embs)

        for agent in self.agents:
            agent.final_output_text = agent.assistant_output_text[-1]
            agent.history           = agent.history_msgs

        return time.perf_counter() - start_time
