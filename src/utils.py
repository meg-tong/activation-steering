import json
import os
from typing import List


def load_json_line(line: str, i: int):
    try:
        return json.loads(line)
    except:
        raise ValueError(f"Error in line {i+1}\n{line}")


def load_from_jsonl(file_name: str):
    with open(file_name, "r") as f:
        data = [load_json_line(line, i) for i, line in enumerate(f)]
    return data


def save_to_jsonl(data: List, file_name: str, overwrite: bool = True) -> None:
    if not overwrite and os.path.exists(file_name):
        print(f"{file_name} was not saved as it already exists.")
        return

    with open(file_name, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")