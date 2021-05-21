"""
Holds project settings.
"""
import os

import appdirs

base_directory = appdirs.user_cache_dir(appname="classicdata")
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 0

os.makedirs(base_directory, exist_ok=True)
