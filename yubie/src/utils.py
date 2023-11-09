import json
import os

from glob import glob
from pathlib import Path

cwd = Path(__file__).parent
config_path = cwd.joinpath('../../config.json')
test_folder_path = cwd.joinpath('../test')

with open(config_path.absolute(), 'r') as file:
    config = json.load(file)


def check_for_custom_flag(args):
    args = args[1:]
    return '-c' in args or '--custom' in args


def get_profile_config(args):
    command = args[0].split('/')[-1]
    user = config.get('user', None)
    profiles = config.get('profiles', None)
    if not user or not profiles:
        return None
    arguments = profiles[user].get(command, {})
    return arguments


def get_test_files():
    test_files = glob('*.test.py', root_dir=test_folder_path.absolute())
    return [
        file_path.replace('.test.py', '')
        for file_path in test_files
    ]

def get_sdk_mapping():
    def get_mapping_from_folder_path(folder_path):
        with open(cwd.joinpath(folder_path,'mapping.json').absolute(), 'r') as file:
            return json.load(file)
    sdk_paths = glob('../sdk/*', root_dir=cwd.absolute())
    mappings = {
        Path(sdk_path).name : get_mapping_from_folder_path(sdk_path)
        for sdk_path in sdk_paths
    }
    return mappings