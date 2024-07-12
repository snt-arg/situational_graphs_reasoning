import json
import os


def get_config(config_name):
    # Determine the path to the JSON file
    config_path = os.path.join(os.path.dirname(__file__), f'{config_name}.json')

    # Load the JSON configuration
    with open(config_path, 'r') as config_file:
        config_data = json.load(config_file)

    return config_data