import os
from typing import List, Optional
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict
import logging

import src.model
import src.utils


hf_token = os.environ["HF_TOKEN"]
BIASED_ANSWER_KEY = "biased_answer"
UNBIASED_ANSWER_KEY = "unbiased_answer"
QUESTION_KEY = "question"
CHOICES_KEY = "choices"
INDEX_TO_LETTER = {0: " (A)", 1: " (B)", 2: " (C)"}


def get_bbq_dataset(configs: List[str], 
                    filename: str = "data/bbq_dataset.jsonl", 
                    question_polarity: Optional[str] = "neg", 
                    context_condition: Optional[str] = None):
    """
    Get original BBQ dataset from huggingface and save it to a jsonl file.
    """
    logging.info("Getting BBQ dataset from huggingface")
    dfs = []
    for config in configs: 
        dataset = load_dataset("heegyu/bbq", config)
        df = pd.DataFrame(dataset['test']) # pyright: ignore
        if question_polarity is not None:
            df = df[df['question_polarity'] == question_polarity]
        if context_condition is not None:
            df = df[df['context_condition'] == context_condition]
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_json(filename, orient="records", lines=True)


def format_group(group: str):
    """
    Some groups are essentially the same but have different names. 
    This function formats them to be the same.
    """
    group = group.lower().replace(" ", "")
    group = group.replace("transgenderwomen", "transgenderwoman")
    if group == "woman" or group == "girl":
        group = "f"
    if group == "man" or group == "boy":
        group = "m"
    return group


def get_biased_answer(d: Dict):
    """
    We get the answer that matches the stereotyped group.
    """
    for ans, options in d['answer_info'].items():
        for o in options:    
            groups = [o] if "-" not in o else o.split("-")
            for g in groups:
                if format_group(d["additional_metadata"]["stereotyped_groups"][0]) == format_group(g):
                    return INDEX_TO_LETTER[int(ans[-1])]
    return None

    
def generate_bias_dataset(bbq_dataset_filename: str = "data/bbq_dataset.jsonl",
                          bias_dataset_filename: str = "data/bias_dataset.jsonl"):
    """
    Generate an alternative version of the original BBQ dataset where
    each question has a binary option only.
    """
    logging.info("Generating bias dataset from BBQ dataset")
    bbq_dataset = src.utils.load_from_jsonl(bbq_dataset_filename)
    bias_dataset = []
    for d in bbq_dataset:
        biased_answer = get_biased_answer(d)
        unbiased_answer = INDEX_TO_LETTER[d["label"]]
        question = d['context'] + " " + d['question']
        choices =  "\n\nChoices:" + "\n (A) " + d["ans0"] + "\n (B) " + d["ans1"] + "\n (C) " + d["ans2"] + "\n\nAnswer:"
        if biased_answer is not None and biased_answer != unbiased_answer:
            bias_dataset.append({BIASED_ANSWER_KEY: biased_answer, 
                                 UNBIASED_ANSWER_KEY: unbiased_answer, 
                                 QUESTION_KEY: question,
                                 CHOICES_KEY: choices})
    src.utils.save_to_jsonl(bias_dataset, bias_dataset_filename)
         

class ComparisonDataset(Dataset):
    def __init__(self, data, system_prompt):
        logging.info("Loading comparison dataset")
        self.data = data
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    @classmethod
    def from_jsonl(cls, filename, system_prompt):
        return cls(src.utils.load_from_jsonl(filename), system_prompt)
    
    @classmethod
    def from_bbq_config(cls, configs: List[str], 
                        system_prompt: str,
                        question_polarity: Optional[str] = "neg",
                        context_condition: Optional[str] = None,
                        bbq_filename: str = "data/bbq_dataset.jsonl", 
                        bias_filename: str ="data/bias_dataset.jsonl",
                        overwrite: bool = False):
        if overwrite or not os.path.exists(bbq_filename):
            get_bbq_dataset(configs, bbq_filename, question_polarity, context_condition)
        else:
            logging.info(f"Skipping getting BBQ dataset as {bbq_filename} already exists")
        if overwrite or not os.path.exists(bias_filename):
            generate_bias_dataset(bbq_filename, bias_filename)
        else:
            logging.info(f"Skipping generating bias dataset as {bias_filename} already exists")
        return cls.from_jsonl(bias_filename, system_prompt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        b_text = item[BIASED_ANSWER_KEY]
        ub_text = item[UNBIASED_ANSWER_KEY]
        q_text = item[QUESTION_KEY] + item[CHOICES_KEY]
        # NOTE: Matches original but maybe we shouldn't be using .strip()?
        b_tokens = src.model.prompt_to_tokens(self.tokenizer, self.system_prompt, instruction=q_text, model_output=b_text.strip())
        ub_tokens = src.model.prompt_to_tokens(self.tokenizer, self.system_prompt, instruction=q_text, model_output=ub_text.strip())
        return b_tokens, ub_tokens