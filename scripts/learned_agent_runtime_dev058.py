"""Exported JSONL inference client. Requires only PyTorch and agent_model.py.

Each request supplies new public `events`, `candidates`, `valid`, `queries`,
and `query_valid`. Optional `reset` starts a new opaque tensor memory.
The returned candidate index is the neural policy's choice, including stop.
This process does not know candidate names, entity identities or world laws.
"""
import json
from pathlib import Path
import sys
import torch
from agent_model import AgentConfig, LearnedAgent


def main():
    root = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = LearnedAgent(AgentConfig(**json.loads((root / 'config.json').read_text()))).eval()
    model.load_state_dict(torch.load(root / 'agent.pt', map_location='cpu', weights_only=True))
    state = model.initial(1)
    with torch.inference_mode():
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('reset', False):
                state = model.initial(1)
            writes = 0
            for event in request['events']:
                state, trace = model.observe(state, torch.tensor([event], dtype=torch.float32))
                writes += trace['writes']
            values = [torch.tensor([request[name]], dtype=dtype) for name, dtype in (
                ('candidates', torch.float32), ('valid', torch.bool),
                ('queries', torch.float32), ('query_valid', torch.bool))]
            result = model(state, *values)
            response = {'index': int(result['logits'].argmax(-1)),
                        'logits': result['logits'][0].tolist(),
                        'completion_probability': float(result['completion'].sigmoid()[0]),
                        'writes': writes, 'work': model.last_work}
            print(json.dumps(response), flush=True)


if __name__ == '__main__':
    main()
