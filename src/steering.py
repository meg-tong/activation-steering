import torch
import os
from tqdm import tqdm


def generate_vectors(model, dataset, 
                                       directory: str = "data/vectors", 
                                       start_layer=6, end_layer=20):
    layers = list(range(start_layer, end_layer + 1))
    diffs = dict([(layer, []) for layer in layers])
    model.set_save_internal_decodings(False)
    model.reset_all()
    for b_tokens, ub_tokens in tqdm(dataset, desc="Processing prompts"):
        b_tokens = b_tokens.to(model.device)
        ub_tokens = ub_tokens.to(model.device)
        model.get_logits(b_tokens)
        for layer in layers:
            s_activations = model.get_last_activations(layer)
            s_activations = s_activations[0, -2, :].detach().cpu()
            diffs[layer].append(s_activations)
        model.get_logits(ub_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            diffs[layer][-1] -= n_activations
    for layer in layers:
        diffs[layer] = torch.stack(diffs[layer])
        torch.save(diffs[layer], os.path.join(directory, f"diff_layer_{layer}.pt"))
        vec = diffs[layer].mean(dim=0)
        torch.save(vec, os.path.join(directory, f"vector_layer_{layer}.pt"))