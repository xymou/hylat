from src.models.agents.nl_agent import NlAgent

class BaseTask:
    def __init__(
        self, 
        agents, 
        dataset, 
        args, 
    ):
        self.agents = agents
        self.dataset = dataset
        self.args = args
        if args.method == "single":
            for agent in self.agents:
                assert type(agent) == NlAgent
    
    def run(data):
        assert False, "Not implemented"