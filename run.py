import os
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

from src.dataset import ComparisonDataset
from src.steering import generate_vectors, get_steered_outputs
from src.model import Llama7BChatHelper
from src.utils import load_from_jsonl


hf_token = os.environ["HF_TOKEN"]
system_prompt = "You are a concise assistant who answers every question."
model = Llama7BChatHelper(token=hf_token, system_prompt=system_prompt)

"""
Generate bias dataset from BBQ dataset of the form:
{"question": "<context><question>", "choices": "<choices>", 
 "biased_answer": "<biased_answer>", "unbiased_answer": "<unbiased_answer>"}
"""
train_bias_comparison_dataset = ComparisonDataset.from_bbq_config(
    ["Race_ethnicity", "Nationality"], 
    question_polarity="neg",
    system_prompt=system_prompt,
    bbq_filename="data/bbq_dataset.jsonl",
    bias_filename="data/bias_dataset.jsonl")

test_bias_comparison_dataset = ComparisonDataset.from_bbq_config(
    configs=["Age", "Disability_status", "Gender_identity", "Physical_appearance", "Religion", "SES", "Sexual_orientation"],
    question_polarity="neg",
    system_prompt=system_prompt,
    bbq_filename="data/bbq_dataset_test.jsonl",
    bias_filename="data/bias_dataset_test.jsonl")

"""
Use the train bias dataset to generate vectors
"""
generate_vectors(model, 
                 train_bias_comparison_dataset, 
                 directory="data/vectors", 
                 overwrite=False,
                 start_layer=2, end_layer=29)

"""
Find the best vector by searching over layers & multipliers
"""
test_dataset = load_from_jsonl("data/bias_dataset_test.jsonl")[::37]
for multiplier in [2]:
    for layer in [12]:
        results = []
        for t in tqdm(test_dataset):
            prompt = t["question"] + t["choices"].replace("\n\nChoices:\n", "").replace("\n\nAnswer:", "").replace("\n", "")
            outputs = get_steered_outputs(model, 
                                          prompt=prompt,
                                          layer=layer, 
                                          multiplier=multiplier,
                                          start=" The answer is (",
                                          split=" The answer is (",
                                          max_length=4)
            result = {key: outputs[key][0] == t["biased_answer"][2] for key in outputs}
            results.append(result)
        biased_fraction = {key: sum([result[key] for result in results]) / len(results) for key in results[0]}
        logging.info(biased_fraction)