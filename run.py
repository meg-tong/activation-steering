import os
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

from src.dataset import ComparisonDataset
from src.steering import generate_vectors, get_steered_outputs, get_vector
from src.model import Llama7BChatHelper
from src.utils import load_from_jsonl, save_to_jsonl


hf_token = os.environ["HF_TOKEN"]
system_prompt = "You are a concise assistant who answers every question."
model = Llama7BChatHelper(token=hf_token, system_prompt=system_prompt)

"""
Generate bias dataset from BBQ dataset of the form:
{"question": "<context><question>", "choices": "<choices>", 
 "biased_answer": "<biased_answer>", "unbiased_answer": "<unbiased_answer>"}

# train: ["Race_ethnicity", "Nationality"]
# test: ["Age", "Disability_status", "Gender_identity", "Physical_appearance", "Religion", "SES", "Sexual_orientation"],
# all: ["Race_ethnicity", "Nationality", "Age", "Disability_status", "Gender_identity", "Physical_appearance", "Religion", "SES", "Sexual_orientation"]
"""
train_bias_comparison_dataset = ComparisonDataset.from_bbq_config(
    ["Race_ethnicity", "Nationality"], 
    question_polarity="neg",
    system_prompt=system_prompt,
    bbq_filename="data/bbq_dataset_train.jsonl",
    bias_filename="data/bias_dataset_train.jsonl")

"""
Use the train bias dataset to generate vectors
"""
vectors_directory = "data/vectors_train"
generate_vectors(model, train_bias_comparison_dataset,
                 directory=vectors_directory, overwrite=False,
                 start_layer=2, end_layer=29)

"""
Find the best vector by searching over layers & multipliers using the test set
"""
test_dataset = load_from_jsonl("data/bias_dataset_test.jsonl")

outputs_directory = "data/outputs"

for multiplier in [2]:
    for layer in [12]:
        vector = get_vector(layer, vectors_directory)
        all_outputs = []
        for t in tqdm(test_dataset):
            prompt = t["question"] + t["choices"].replace("\n\nChoices:\n", "").replace("\n\nAnswer:", "").replace("\n", "")
            outputs = get_steered_outputs(model, prompt=prompt, layer=layer, vector=vector, multiplier=multiplier,
                                          start=" The answer is (", split=" The answer is (", max_length=4)
            scores = {"score_" + key: outputs[key][0] == t["biased_answer"][2] for key in outputs}
            all_outputs.append({**scores, **outputs})
            if len(all_outputs) % 100 == 0 or len(all_outputs) == len(test_dataset):
                logging.info({key: sum([outputs[key] for outputs in all_outputs]) / len(all_outputs) for key in all_outputs[0] if key.startswith("score")})
        save_to_jsonl(all_outputs, f"data/outputs/output_layer_{layer}_multiplier_{multiplier}.jsonl")