import json


def jopen(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        j = json.load(f)

    return j