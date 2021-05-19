"""
Holds project settings.
"""
import os

import appdirs

base_directory = appdirs.user_cache_dir(appname="classic-data")

os.makedirs(base_directory, exist_ok=True)
