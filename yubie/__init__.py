import importlib
import re

from glob import glob
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from InquirerPy import inquirer
from yubie.src.decorators import extract_config
from yubie.src.utils import get_sdk_mapping, get_test_files
from .main import (
    Topics as Topics,
    vision_module as VisionModule,
    pubsub
)

load_dotenv(find_dotenv())

modules = {}
cwd = Path(__file__).parent


def init_all_modules():
    global modules
    prefix = 'yubie.modules.'

    modules_folder_path = cwd.joinpath('./modules')
    all_files = [file.replace('.py', '').replace('/', '.')
                 for file in glob('**/*.py', root_dir=modules_folder_path)]
    modules = {
        file: importlib.import_module(prefix + file)
        for file in all_files
    }


@extract_config
def dev(has_custom_flag, profile_config):
    # Here multiple modules can be called
    print("Initialized Dev Mode")


@extract_config
def test(has_custom_flag, profile_config):
    if has_custom_flag or not profile_config:
        profile_config['test-suite'] = inquirer.fuzzy(
            message="Select test suite to execute",
            choices=get_test_files(),
            max_height=200
        ).execute()
    test_module = "yubie.test."+profile_config['test-suite']
    try:
        # Run all the modules before testing
        init_all_modules()
        module = importlib.import_module(test_module)
        module.main()
    except ModuleNotFoundError:
        print(f"Module '{test_module}' not found.")


@dataclass
class SpotOptions:
    hostname = '192.168.80.3'
    verbose = False


@extract_config
def collect_image_data(has_custom_flag, profile_config):
    sdk_mappings = get_sdk_mapping()
    if has_custom_flag or not profile_config:
        profile_config['sdk'] = inquirer.fuzzy(
            message="Select the Robot SDK to connect",
            choices=sdk_mappings.keys(),
            max_height=200
        ).execute()

    config = sdk_mappings[profile_config['sdk']]
    entry_points = config.get('entry_points', {})
    module_for_image_data = entry_points.get('collect_image_data', None)
    if not module_for_image_data:
        print(
            f"Module for 'collect_image_data' is not provided in entry_points of {profile_config['sdk']}/mapping.json")

    def prepare(module_name):
        module_name = re.sub(r'(\.py$|^\./)', '', module_name)
        return module_name.replace('/', '.')

    try:
        init_all_modules()
        actual_module = 'yubie.sdk.' + \
            profile_config['sdk'] + '.' + prepare(module_for_image_data)
        module = importlib.import_module(actual_module)

        options = SpotOptions()
        module.main(options)
    except ModuleNotFoundError:
        print(f"Module '{actual_module}' not found.")
