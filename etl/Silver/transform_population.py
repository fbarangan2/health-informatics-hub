# test
# using the config added in the root directory so it does not depend on hardcode variable

from utils.config_loader import load_config

config = load_config()

bronze_path = config["paths"]["bronze"]
silver_path = config["paths"]["silver"]

population_file = config["datasets"]["population"]

print(f"Reading from {bronze_path}/{population_file}")
