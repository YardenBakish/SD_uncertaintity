import json
import shutil
import os

def update_json(filename, d):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    for k in d:
        data[k] = d[k]
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    


def collect_and_merge_results(root_dir):
    root_dir = os.path.abspath(root_dir)
    global_res_path = os.path.join(root_dir, "res.json")

    collected = {}

    # 1. Collect subdirectory results
    for name in os.listdir(root_dir):
        subdir = os.path.join(root_dir, name)
       
        if not os.path.isdir(subdir):
            continue

        res_path = os.path.join(subdir, "res.json")
        if os.path.isfile(res_path):
            with open(res_path, "r") as f:
                collected[name] = json.load(f)

    # 2. Load global res.json if exists
    if os.path.isfile(global_res_path):
        with open(global_res_path, "r") as f:
            global_res = json.load(f)
    else:
        global_res = {}

    # 3. Merge (non-destructive update)
    for method, method_dict in collected.items():
        if method not in global_res:
            global_res[method] = {}

        for k, v in method_dict.items():
            global_res[method][k] = v

    # 4. Write updated global res.json
    with open(global_res_path, "w") as f:
        json.dump(global_res, f, indent=2)

    # 5. Delete all subdirectories
    for name in os.listdir(root_dir):
        subdir = os.path.join(root_dir, name)
        
        if os.path.isdir(subdir):
            shutil.rmtree(subdir)