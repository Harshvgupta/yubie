import importlib

from glob import glob
from pathlib import Path
from InquirerPy import inquirer
from yubie.src.decorators import extract_config
from yubie.src.utils import get_test_files
from .main import (
    Topics as Topics,
    vision_module as VisionModule,
    pubsub
)

modules = {}


def init_all_modules():
    global modules
    prefix = 'yubie.modules.'
    cwd = Path(__file__).parent
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
