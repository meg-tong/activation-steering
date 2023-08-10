import torch
import os
from tqdm import tqdm
import logging

from src.model import Llama7BChatHelper
from src.dataset import ComparisonDataset


def generate_vectors(model: Llama7BChatHelper, 
                     dataset: ComparisonDataset,
                     directory: str = "data/vectors", 
                     overwrite: bool = False,
                     start_layer=6, end_layer=20):
    layers = list(range(start_layer, end_layer + 1))
    if not overwrite:
        layers = [layer for layer in layers if not os.path.exists(os.path.join(directory, f"vector_layer_{layer}.pt"))]
    if len(layers) == 0:
        logging.info("Skipping generating vectors as they already exist")
        return
    diffs = dict([(layer, []) for layer in layers])
    model.set_save_internal_decodings(False)
    model.reset_all()
    for b_tokens, ub_tokens in tqdm(dataset, desc="Generating vectors from dataset"):
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
        
        
def get_vec(layer, directory="data/vectors"):
    return torch.load(os.path.join(directory, f"vector_layer_{layer}.pt"))        

        
def get_steered_outputs(model: Llama7BChatHelper,
                        prompt: str,
                        layer: int,
                        multiplier: int,
                        max_length: int = 20,
                        first_token_only=False,
                        start=" The answer is (",
                        split=" The answer is ("):
    
    model.set_only_add_to_first_token(first_token_only)
    vec = get_vec(layer)
    pos_multiplier = multiplier
    neg_multiplier = multiplier * -1.0
    model.set_save_internal_decodings(False)
    
    model.reset_all()
    model.set_add_activations(layer, pos_multiplier * vec.cuda())
    answer_plus_bias = model.generate_text(prompt, model_output=start, max_length=max_length)
    
    model.reset_all()
    model.set_add_activations(layer, neg_multiplier * vec.cuda())
    answer_minus_bias = model.generate_text(prompt, model_output=start, max_length=max_length)
    
    model.reset_all()
    answer = model.generate_text(prompt, model_output=start, max_length=max_length)

    if split != "":
        answer_plus_bias = answer_plus_bias.split(split)[-1]
        answer_minus_bias = answer_minus_bias.split(split)[-1]
        answer = answer.split(split)[-1]
        
    return {"answer": answer, "answer+bias": answer_plus_bias, "answer-bias": answer_minus_bias}