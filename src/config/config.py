import yaml


def get_config(path="./config/neo4j.yaml"):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
