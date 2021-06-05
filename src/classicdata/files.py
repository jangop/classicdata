"""
Downloads and keeps files locally.
"""
import hashlib
import os
from urllib.request import urlretrieve

from loguru import logger


def compute_hash(path):
    """

    :param path:
    :return:
    """
    file_hash = hashlib.blake2b()
    with open(path, "rb") as handle:
        while chunk := handle.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def provide_file(
    url: str, root_directory: str, sub_directory: str, expected_hash: str = None
) -> str:
    """
    Downloads if necessary and checks a file.
    :param url: Where the file can be downloaded from.
    :param root_directory: Where files are stored locally.
    :param sub_directory: Where this file is stored below the root directory.
    :param expected_hash: Expected hash of the file.
    :return: Local path to the downloaded file.
    """
    # Prepare paths.
    filename = os.path.basename(url)
    directory = os.path.join(root_directory, sub_directory)
    os.makedirs(directory, exist_ok=True)
    local_path = os.path.join(directory, filename)

    # Download file if it does not exist.
    if not os.path.exists(local_path):
        urlretrieve(url, local_path)

    # Check hash.
    actual_hash = compute_hash(local_path)
    if expected_hash is None:
        logger.critical(f"hash for {local_path}: {actual_hash}")
    elif expected_hash != actual_hash:
        raise RuntimeError(
            f"`{local_path}` hashes to "
            f"`{actual_hash}` instead of "
            f"`{expected_hash}`"
        )
    return local_path
