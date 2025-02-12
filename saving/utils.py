import json


def json_save(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f,indent=4)