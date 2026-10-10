"""Resolve local configuration paths without importing the ML runtime."""
from os import fspath, path as osp


def resolve_project_path(value, project_root):
    """Resolve relative paths against the repository and preserve absolute paths."""
    if value is None:
        return None
    value = osp.expanduser(fspath(value))
    if not osp.isabs(value):
        value = osp.join(fspath(project_root), value)
    return osp.abspath(value)
