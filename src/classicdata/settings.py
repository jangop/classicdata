"""
Holds project settings.
"""
import os

import appdirs

base_directory = appdirs.user_cache_dir(appname="classicdata")
default_test_size = 0.2
default_random_state = 0

os.makedirs(base_directory, exist_ok=True)
