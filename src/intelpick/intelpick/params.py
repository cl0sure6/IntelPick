"""Read the installed config/intelpick.yaml from the command-line tools (no ROS node needed)."""

import os

import yaml


def installed_params(*sections):
    """ros__parameters of each named section ('/**', 'arm', 'detector', ...), {} if missing.

    Tools use these as their defaults, so values tuned with a tool go straight back into the
    same file the nodes read.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('intelpick'), 'config', 'intelpick.yaml')
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception:  # not built / not sourced: callers fall back to built-in defaults
        data = {}
    return tuple((data.get(s) or {}).get('ros__parameters', {}) for s in sections)
