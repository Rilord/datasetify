from pathlib import Path
import json


def json_load(file="annotations.json"):
    assert Path(file).suffix in (
        ".json"
    ), f"Attempting to load non-JSON file {file} with json_load()"
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

        return data
