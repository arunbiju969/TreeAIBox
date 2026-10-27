"""
Model storage directories, model zoo discovery, and download utilities for TreeAIBox.
"""

import os
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import requests


def get_model_storage_dir() -> str:
    """Create and return the model storage directory path."""
    if os.name == 'nt':  # Windows
        appdata_dir = Path(os.getenv('LOCALAPPDATA', ''))
    else:  # macOS, Linux
        appdata_dir = Path.home() / '.local' / 'share'

    model_dir = appdata_dir / 'CloudCompare' / 'TreeAIBox' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir)


def get_log_storage_dir() -> str:
    """Create and return the log storage directory path."""
    if os.name == 'nt':  # Windows
        appdata_dir = Path(os.getenv('LOCALAPPDATA', ''))
    else:  # macOS, Linux
        appdata_dir = Path.home() / '.local' / 'share'

    log_dir = appdata_dir / 'CloudCompare' / 'TreeAIBox' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def check_model_exists(model_name: str, model_dir: Optional[str] = None) -> bool:
    """Check if the model .pth file exists locally."""
    storage_dir = model_dir or get_model_storage_dir()
    file_path = os.path.join(storage_dir, f"{model_name}.pth")
    return os.path.isfile(file_path)


def load_model_zoo(zoo_path: str) -> List[Dict[str, Any]]:
    """Load model entries from model_zoo.json."""
    if not os.path.exists(zoo_path):
        with open(zoo_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return []

    try:
        with open(zoo_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load model zoo from '{zoo_path}': {e}")
        return []


def download_model_file(
    url: str,
    local_path: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> bool:
    """
    Download a remote model file with streamed chunks and progress notification.

    Args:
        url: Remote model download URL.
        local_path: Destination local file path.
        progress_callback: Optional callable accepting integer percent (0-100).

    Returns:
        True on successful download and file placement, False otherwise.
    """
    temp_path = local_path + ".temp"
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and progress_callback is not None:
                        percent = int((downloaded / total_size) * 100)
                        progress_callback(percent)

        if os.path.exists(local_path):
            os.remove(local_path)
        os.rename(temp_path, local_path)
        return True

    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise e
