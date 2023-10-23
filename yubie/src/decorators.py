import sys
from yubie.src.utils import check_for_custom_flag, get_profile_config


def extract_config(func):
    def wrapper():
        has_custom_flag = check_for_custom_flag(sys.argv)
        profile_config = get_profile_config(sys.argv)
        func(has_custom_flag, profile_config)
    return wrapper

