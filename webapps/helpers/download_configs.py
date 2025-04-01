# -*- coding: utf-8 -*-
import os

import requests
from sinapsis_core.utils.logging_utils import sinapsis_logger

REPO_URL = "https://api.github.com/repos/Peterande/D-FINE/contents/configs"
HEADERS = {"Accept": "application/vnd.github.v3+json"}


def download_configs_folder(url: str = REPO_URL, path: str = "configs") -> None:
    """Recursively downloads all files from a GitHub repository folder.

    Args:
        url (str, optional): GitHub API URL of the folder to download. Defaults to REPO_URL.
        path (str, optional): Local path where files will be saved. Defaults to "configs".
    """
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        sinapsis_logger.error(f"Failed to fetch folder contents. Status code: {response.status_code}")
        return

    os.makedirs(path, exist_ok=True)

    for item in response.json():
        if item["type"] == "file":
            file_name = item["name"]
            file_path = os.path.join(path, file_name)
            if os.path.exists(file_path):
                continue

            file_url = item["download_url"]
            file_response = requests.get(file_url)
            if file_response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(file_response.content)
            else:
                sinapsis_logger.error(f"Failed to download: {file_name}")
        elif item["type"] == "dir":
            subfolder_name = item["name"]
            subfolder_path = os.path.join(path, subfolder_name)
            download_configs_folder(item["url"], subfolder_path)
