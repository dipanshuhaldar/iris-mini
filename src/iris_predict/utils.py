import yaml
def load_cfg(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
