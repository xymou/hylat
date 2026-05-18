import os
import sys
import json
import copy
import time
import torch
import hashlib
import argparse
from tqdm import tqdm
from termcolor import colored

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.insert(0, project_root)

from src.data import (
    GSM8K, MMLU, MedQA, GPQA, ARC, StrategyQA, CommonsenseQA, SocialIQA,
    WikiMultihopQA, StrategyQA, ComplexWebQuestions, QuasarT,
    HotpotQA, FEVER, MMLUDebug, WorldTree, PubMed, OpenbookQA, QED, DebugMCData,
    Partisan, Opinion, TrustGame
)

from src.models.agents import NlAgent, CipherAgent, SDEAgent, LatentStage2Agent, PureLatentAgent
from src.root_path import ROOT_PATH, DATA_ROOT_PATH
from src.utils import load_model, load_only_generation_config, load_latent_model, count_cipher, count_token, generate_directed_topology, count_trust_game
from src.tasks import method_to_task_cls, update_hidden_dim


def generate_args_hash(
    setting: dict, 
    ignore_keys=["config_file", "sample", "redo", "sde_layer_config", "output_dir"],
):
    setting = {k: v for k, v in setting.items() if k not in ignore_keys}
    setting = json.dumps(setting, sort_keys=True)
    hash_output = hashlib.md5(setting.encode()).hexdigest()
    return hash_output


def main(args):
    if args.dataset == "gsm8k":
        dataset = GSM8K(data_root_path=DATA_ROOT_PATH)
    elif args.dataset.startswith("mmlu_"):
        field_name = args.dataset.split("mmlu_")[1]
        dataset = MMLU(data_root_path=DATA_ROOT_PATH, field_name=field_name)
    elif args.dataset == "medqa":
        dataset = MedQA(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == "gpqa":
        dataset = GPQA(data_root_path=DATA_ROOT_PATH)
    elif args.dataset.startswith("arc"):
        field_name = args.dataset.split("arc_")[1]
        dataset = ARC(data_root_path=DATA_ROOT_PATH, field_name=field_name)
    elif args.dataset == "2wqa":
        dataset = WikiMultihopQA(data_root_path=DATA_ROOT_PATH, retrieval_topk=args.retrieval_topk)
    elif args.dataset == "cwq":
        dataset = ComplexWebQuestions(data_root_path=DATA_ROOT_PATH, retrieval_topk=args.retrieval_topk)
    elif args.dataset == "quasart":
        dataset = QuasarT(data_root_path=DATA_ROOT_PATH, retrieval_topk=args.retrieval_topk)
    elif args.dataset == "hotpotqa":
        dataset = HotpotQA(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == "fever":
        dataset = FEVER(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == "debug":
        dataset = MMLUDebug(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == 'commonsense':
        dataset = CommonsenseQA(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == "strategy":
        dataset = StrategyQA(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == 'socialiqa':
        dataset = SocialIQA(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == 'worldtree':
        dataset = WorldTree(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == 'pubmed':
        dataset = PubMed(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == 'open_book_qa':
        dataset = OpenbookQA(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == 'qed':
        dataset = QED(data_root_path=DATA_ROOT_PATH)      
    elif args.dataset == "debug_commonsense":
        dataset = DebugMCData(data_root_path=DATA_ROOT_PATH, subset_path="commonsense_subset.jsonl")
    elif args.dataset == "debug_medqa":
        dataset = DebugMCData(data_root_path=DATA_ROOT_PATH, subset_path="med_qa_subset.jsonl")   
    elif args.dataset == "debug_open_book_qa":
        dataset = DebugMCData(data_root_path=DATA_ROOT_PATH, subset_path="open_book_qa_subset.jsonl")     
    elif args.dataset == "debug_pubmed_qa":
        dataset = DebugMCData(data_root_path=DATA_ROOT_PATH, subset_path="pubmed_qa_subset.jsonl")         
    elif args.dataset == "debug_socialiqa":
        dataset = DebugMCData(data_root_path=DATA_ROOT_PATH, subset_path="socialiqa_subset.jsonl")       
    elif args.dataset == "debug_worldtree":
        dataset = DebugMCData(data_root_path=DATA_ROOT_PATH, subset_path="worldtree_subset.jsonl")  
    elif args.dataset == "partisan":
        dataset = Partisan(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == "opinion":
        dataset = Opinion(data_root_path=DATA_ROOT_PATH)     
    elif args.dataset == "trust_game":
        dataset = TrustGame(data_root_path=DATA_ROOT_PATH)        
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")
    
    # set system prompt
    role_prompt = None
    greedy_generation_config = dict(
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    greedy_cipher_generation_config = [dict(temperature=0.0) for i in range(args.agent_cnt)]
    
    if args.generation_setting == "greedy":
        generation_config = dict(
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        cipher_generation_config = [dict(temperature=0.0) for i in range(args.agent_cnt)]
    else:
        # generation_config = dict()
        default_generation_config = load_only_generation_config(args.model_name_or_path)
        default_generation_config["temperature"]=args.temperature 
        default_generation_config["top_k"] = 20
        default_generation_config["top_p"] = 0.9
        default_generation_config["do_sample"] = True
        generation_config = default_generation_config 
        cipher_generation_config = []
        for i in range(args.agent_cnt):
            cipher_generation_config.append(dict(temperature=default_generation_config["temperature"] * (i + 1) / args.agent_cnt))
    
    # create output file
    setting_dir = generate_args_hash(vars(args), ignore_keys=["output_dir", "resume", "redo", "resume_from", "sample", "test_layer"])
    model_save_name = args.model_name_or_path.split("/")[-1]
    if len(model_save_name) == 0:
        model_save_name = args.model_name_or_path.split("/")[-2]
    if "latent" in args.method: model_save_name+="_"+args.add_name
    
    output_dir = os.path.join(ROOT_PATH, args.output_dir, args.task_type, args.dataset, model_save_name, args.method, setting_dir)
    os.makedirs(output_dir, exist_ok=True)
    if args.redo:
        run_id = len(os.listdir(output_dir))
    else:
        run_id = max(0, len(os.listdir(output_dir)) - 1) 
    output_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as fout:
        json.dump(vars(args), fout, indent=4)
    detail_file = os.path.join(output_dir, "detail.json")
    result_file = os.path.join(output_dir, "result.json")
    print(colored(f"### Output dir: {output_dir} ###", "green"))

    if os.path.exists(detail_file):
        with open(detail_file, "r") as fin:
            details = json.load(fin)
    else:
        details = []
    start_from = len(details)
    
    if args.task_type == "trust_game":
        if (args.sample != -1 and start_from >= args.sample) or \
        (args.sample == -1 and start_from == len(dataset.dataset)): # if enough, skip, generate result
            if args.sample != -1:
                details = details[:args.sample]
            print(colored(f"### Already finished {start_from} samples, skip ###", "green"))
            print(colored(f"### Total sampled {len(details)} ###", "green"))
            task_cls = method_to_task_cls(task_type=args.task_type, method=args.method)
            run_task = task_cls([], dataset, args)
            if "latent" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
            elif "pure_text" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
                model = model.codi
            else:
                model, tokenizer, model_config = load_model(args.model_name_or_path, args.method)       
        else:
            if "latent" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
            elif "pure_text" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
                model = model.codi
            else:
                model, tokenizer, model_config = load_model(args.model_name_or_path, args.method)
            update_hidden_dim(model_config.hidden_size)
            agents = []
            for idx in range(args.agent_cnt):   # agent_cnt 设为 2
                if args.method == "nl": 
                    agent_cls = NlAgent
                elif args.method == "cipher":
                    agent_cls = CipherAgent
                elif args.method == "sde":
                    agent_cls = SDEAgent
                elif args.method == "autoform":
                    agent_cls = NlAgent
                elif args.method == "latent_stage2":
                    agent_cls = LatentStage2Agent
                elif args.method == "pure_latent":
                    agent_cls = PureLatentAgent
                elif args.method == "pure_text":
                    agent_cls = NlAgent
                else:
                    raise ValueError(f"Invalid method {args.method}")
                agents.append(agent_cls(
                    engine_model_name_or_path=args.model_name_or_path,
                    engine_model=model,
                    engine_tokenizer=tokenizer,
                    generation_configs=cipher_generation_config[idx] if args.method == "cipher" else generation_config,
                    max_new_tokens=args.max_new_tokens,
                    role_prompt=None,
                )) 
            dataset.sample(sample_cnt=args.sample)
            task_cls = method_to_task_cls(task_type=args.task_type, method=args.method)
            run_task = task_cls(agents, dataset, args)
            pbar = tqdm(total=len(dataset.dataset) - len(details))
            for data in dataset.dataset[start_from:]:
                detail = run_task.run(data)
                detail["test_id"] = data["test_id"]
                details.append(detail)
                pbar.update(1)
                with open(detail_file, "w") as fout:
                    json.dump(details, fout, indent=4)
            pbar.close()

            res = vars(args)
            res.update(run_task.generate_result(details))                            
    
    elif args.dataset in ["partisan", "opinion"]:
        if (args.sample != -1 and start_from >= args.sample) or \
        (args.sample == -1 and start_from == len(dataset.dataset)): # if enough, skip, generate result
            if args.sample != -1:
                details = details[:args.sample]
            print(colored(f"### Already finished {start_from} samples, skip ###", "green"))
            print(colored(f"### Total sampled {len(details)} ###", "green"))
            task_cls = method_to_task_cls(task_type=args.task_type, method=args.method)
            run_task = task_cls([], dataset, args)
            if "latent" in args.method or "short" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
            elif "pure_text" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
                model = model.codi
            else:
                model, tokenizer, model_config = load_model(args.model_name_or_path, args.method)        
        else:
            if "latent" in args.method or "short" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
            elif "pure_text" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
                model = model.codi
            else:
                model, tokenizer, model_config = load_model(args.model_name_or_path, args.method)
            update_hidden_dim(model_config.hidden_size)
            need_agent_cnt = args.agent_cnt
            agent_info_list = json.load(open(args.agent_info_list,"r"))
            agents = []
            for idx in range(need_agent_cnt):
                if args.method == "nl": 
                    agent_cls = NlAgent
                elif args.method == "cipher":
                    agent_cls = CipherAgent
                elif args.method == "sde":
                    agent_cls = SDEAgent
                elif args.method == "autoform":
                    agent_cls = NlAgent
                elif args.method == "latent_stage2":
                    agent_cls = LatentStage2Agent
                elif args.method == "pure_latent":
                    agent_cls = PureLatentAgent
                elif args.method == "pure_text":
                    agent_cls = NlAgent
                else:
                    raise ValueError(f"Invalid method {args.method}")
                agents.append(agent_cls(
                    engine_model_name_or_path=args.model_name_or_path,
                    engine_model=model,
                    engine_tokenizer=tokenizer,
                    generation_configs=cipher_generation_config[idx] if args.method == "cipher" else generation_config,
                    max_new_tokens=args.max_new_tokens,
                    role_prompt=role_prompt,
                    agent_persona = agent_info_list[idx]['agent_persona'],
                    agent_party = agent_info_list[idx]['agent_party'] if "agent_party" in agent_info_list[idx] else ""
                ))
            # load agent network
            if os.path.exists(args.agent_topology):
                agent_topology = json.load(open(args.agent_topology,"r"))
                agent_topology = {int(key):value for key, value in agent_topology.items()}
            else:
                # agent_topology = generate_directed_topology(len(agents), num_neighbors=1)
                agent_topology = None
            args.agent_topology = agent_topology
            dataset.sample(sample_cnt=args.sample)
            task_cls = method_to_task_cls(task_type=args.task_type, method=args.method)
            run_task = task_cls(agents, dataset, args)
            pbar = tqdm(total=len(dataset.dataset) - len(details))
            for data in dataset.dataset[start_from:]:
                detail = run_task.run(data)
                detail["test_id"] = data["test_id"]
                details.append(detail)
                pbar.update(1)
                with open(detail_file, "w") as fout:
                    json.dump(details, fout, indent=4)
            pbar.close()
    else:
        if (args.sample != -1 and start_from >= args.sample) or \
        (args.sample == -1 and start_from == len(dataset.dataset)): # if enough, skip, generate result
            if args.sample != -1:
                details = details[:args.sample]
            print(colored(f"### Already finished {start_from} samples, skip ###", "green"))
            print(colored(f"### Total sampled {len(details)} ###", "green"))
            task_cls = method_to_task_cls(task_type=args.task_type, method=args.method)
            run_task = task_cls([], dataset, args)
            if "latent" in args.method or "short" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
            elif "pure_text" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
                model = model.codi
            else:
                model, tokenizer, model_config = load_model(args.model_name_or_path, args.method)        
        else:
            if "latent" in args.method or "short" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
            elif "pure_text" in args.method:
                model, tokenizer, model_config = load_latent_model(args.model_name_or_path, args.conf_path)
                model = model.codi
            else:
                model, tokenizer, model_config = load_model(args.model_name_or_path, args.method)
            update_hidden_dim(model_config.hidden_size)
            need_agent_cnt = args.agent_cnt
            agents = []
            for idx in range(need_agent_cnt):
                if args.method == "nl": 
                    agent_cls = NlAgent
                elif args.method == "cipher":
                    agent_cls = CipherAgent
                elif args.method == "sde":
                    agent_cls = SDEAgent
                elif args.method == "autoform":
                    agent_cls = NlAgent
                elif args.method == "latent_stage2":
                    agent_cls = LatentStage2Agent
                elif args.method == "pure_latent":
                    agent_cls = PureLatentAgent
                elif args.method == "pure_text":
                    agent_cls = NlAgent
                else:
                    raise ValueError(f"Invalid method {args.method}")

                agents.append(agent_cls(
                    engine_model_name_or_path=args.model_name_or_path,
                    engine_model=model,
                    engine_tokenizer=tokenizer,
                    generation_configs=cipher_generation_config[idx] if args.method == "cipher" else generation_config,
                    max_new_tokens=args.max_new_tokens,
                    role_prompt=role_prompt,
                ))

            dataset.sample(sample_cnt=args.sample)
            task_cls = method_to_task_cls(task_type=args.task_type, method=args.method)
            run_task = task_cls(agents, dataset, args)
            pbar = tqdm(total=len(dataset.dataset) - len(details))
        
            for data in dataset.dataset[start_from:]:
                detail = run_task.run(data)
                detail["test_id"] = data["test_id"]            
                details.append(detail)
                pbar.update(1)
                with open(detail_file, "w") as fout:
                    json.dump(details, fout, indent=4)
            pbar.close()
    
    
    res = vars(args)
    res.update(run_task.generate_result(details))
    if args.dataset=="trust_game":
        num_tokens = count_trust_game(details, tokenizer)
    else:
        if args.method in ["cipher","sde"]:
            num_tokens = count_cipher(details, tokenizer)
        else:
            num_tokens = count_token(details, tokenizer)
    
    res["num_tokens"] = num_tokens    
    with open(result_file, "w") as fout:
        json.dump(res, fout, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="hylat/eval/configs/debate.json")
    parser.add_argument("--model_name_or_path", type=str, default="Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", type=str, default="pubmed")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--method", type=str, choices=["latent_stage2", "pure_latent", "pure_text"], default="latent_stage2")
    parser.add_argument("--output_dir", type=str, default="output_debug")
    parser.add_argument("--redo", action="store_true") # if True, do a new run
    
    # args in config file
    parser.add_argument("--task_type", type=str, choices=["ia", "debate", "workflow"])
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--agent_cnt", type=int)
    parser.add_argument("--generation_setting", type=str, choices=["default", "greedy"])
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--prompt_file", type=str)
    parser.add_argument("--retrieval_topk", type=int) # for IA, retrieval passage number
    parser.add_argument("--temperature", type=float, default=0.2)

    # args for sde
    parser.add_argument("--edit_layer_idx", type=str, default=None) # such as "1" or "1,2,3,4"
    parser.add_argument("--sde_layer_config", type=str, default="configs/sde_layer.json") # default setting 
    
    # args for latent
    parser.add_argument("--conf_path", type=str, default="hylat/hylat/configs/test_mas.yaml")
    parser.add_argument("--add_name", type=str, default="") 
    
    args = parser.parse_args()

    if args.config_file is not None:
        try:
            with open(args.config_file, "r") as fin:
                config = json.load(fin)
            if config["task_type"] == "workflow":
                config["max_new_tokens"] = config["max_new_tokens_ma"]  
                del config["max_new_tokens_single"]
                del config["max_new_tokens_ma"]
            for k, v in config.items():
                if k not in vars(args) or getattr(args, k) is None:
                    setattr(args, k, v)
        except:
            assert False, "Invalid config file"

    if args.method == "sde":
        if args.edit_layer_idx is None:
            with open(args.sde_layer_config, "r") as fin:
                cfg = json.load(fin)
                args.edit_layer_idx = cfg[args.model_name_or_path]
        edit_layer_idx = args.edit_layer_idx.split(",")
        edit_layer_idx = [int(x) for x in edit_layer_idx] 
        args.edit_layer_idx = sorted(edit_layer_idx)
    
    if args.task_type == "debate" and args.sample is None:
        if args.dataset in ["gsm8k", "commonsense", "strategy", "socialiqa", "worldtree", "pubmed", "open_book_qa", "qed", "medqa"]:
            args.sample = 300
        elif args.dataset.startswith("arc"):
            args.sample = 300
        else:
            args.sample = -1
    
    assert args.sample is not None

    print(args)
    main(args)