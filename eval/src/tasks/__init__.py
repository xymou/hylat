# from src.tasks.ia import SingleIATask, NlIATask, CipherIATask, SDEIATask, LatentStage1IATask, LatentStage2IATask
from src.tasks.ia_llama import SingleIATask, NlIATask, CipherIATask, SDEIATask, LatentStage1IATask, LatentStage2IATask
from src.tasks.debate import SingleDebateTask, NlDebateTask, CipherDebateTask, SDEDebateTask, LatentStage1DebateTask, LatentStage2DebateTask, \
    ShortAnswerDebateTask, LatentMASDebateTask, PureLatentDebateTask, HybridDebateTask
from src.tasks.workflow import SingleWorkflowTask, NlWorkflowTask, CipherWorkflowTask, SDEWorkflowTask
from src.tasks.utils import update_hidden_dim
from src.tasks.social import SingleSocialTask, NlSocialTask, CipherSocialTask, SDESocialTask, LatentStage1SocialTask, LatentStage2SocialTask, LatentMASSocialTask
from src.tasks.game import (
    NlRepeatedTrustGameTask,
    CipherRepeatedTrustGameTask,
    LatentRepeatedTrustGameTask,
    SDERepeatedTrustGameTask,
)


def method_to_task_cls(task_type, method):
    task_type_to_cls = {
        "debate": {
            "single": SingleDebateTask,
            "nl": NlDebateTask,
            "cipher": CipherDebateTask,
            "sde": SDEDebateTask,
            "autoform": NlDebateTask,
            "ecolang": NlDebateTask,
            "latent_stage1": LatentStage1DebateTask,
            "latent_stage2": LatentStage2DebateTask,
            "short":ShortAnswerDebateTask,
            "latentmas": LatentMASDebateTask,
            "pure_text":NlDebateTask,
            "pure_latent":PureLatentDebateTask,
            "hybrid":HybridDebateTask
        },
        "ia": {
            "single": SingleIATask,
            "nl": NlIATask,
            "cipher": CipherIATask,
            "sde": SDEIATask,
            "autoform": NlIATask,
            "ecolang": NlIATask,
            "latent_stage1": LatentStage1IATask,
            "latent_stage2": LatentStage2IATask,
            "openai":NlIATask
        },
        "workflow": {
            "single": SingleWorkflowTask,
            "nl": NlWorkflowTask,
            "cipher": CipherWorkflowTask,
            "sde": SDEWorkflowTask,
            "autoform": NlWorkflowTask,
        },
        "social":{
            "single": SingleSocialTask,
            "nl": NlSocialTask,
            "cipher": CipherSocialTask,
            "sde": SDESocialTask,
            "autoform": NlSocialTask,
            "ecolang": NlSocialTask,
            "latent_stage1": LatentStage1SocialTask,
            "latent_stage2": LatentStage2SocialTask,
            "latentmas": LatentMASSocialTask            
        },
        "trust_game":{
            "single": NlRepeatedTrustGameTask,
            "nl": NlRepeatedTrustGameTask,
            "cipher": CipherRepeatedTrustGameTask,
            "sde": SDERepeatedTrustGameTask,
            "autoform": NlRepeatedTrustGameTask,
            "ecolang": NlRepeatedTrustGameTask,
            "latent_stage2": LatentRepeatedTrustGameTask,            
        }
    }
    if task_type not in task_type_to_cls:
        raise NotImplementedError(f"Task type {task_type} is not implemented.")
    if method not in task_type_to_cls[task_type]:
        raise NotImplementedError(f"Method {method} is not implemented for task type {task_type}.")
    return task_type_to_cls[task_type][method]