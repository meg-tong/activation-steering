import os

from src.dataset import ComparisonDataset
from src.steering import generate_vectors
from src.model import Llama7BChatHelper


hf_token = os.environ["HF_TOKEN"]
system_prompt = "You are a concise assistant who tries to answer every question."

"""
Generate bias dataset from BBQ dataset of the form:
{"question": "<context><question>", "choices": "<choices>", 
 "biased_answer": "<biased_answer>", "unbiased_answer": "<unbiased_answer>"}
"""

train_bias_dataset = ComparisonDataset.from_bbq_config(
    ["Race_ethnicity", "Nationality"], 
    question_polarity="neg",
    system_prompt=system_prompt,
    bbq_filename="data/bbq_dataset.jsonl",
    bias_filename="data/bias_dataset.jsonl")

test_bias_dataset = ComparisonDataset.from_bbq_config(
    configs=["Age", "Disability_status", "Gender_identity", "Physical_appearance", "Religion", "SES", "Sexual_orientation"],
    question_polarity="neg",
    system_prompt=system_prompt,
    bbq_filename="data/bbq_dataset_test.jsonl",
    bias_filename="data/bias_dataset_test.jsonl")

"""
Use the train bias dataset to generate vectors
"""
model = Llama7BChatHelper(token=hf_token, system_prompt=system_prompt)
generate_vectors(model, 
                 train_bias_dataset, 
                 directory="data/vectors", 
                 start_layer=6, end_layer=20)

