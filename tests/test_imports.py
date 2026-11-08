"""Every module imports cleanly (catches syntax errors in code no other test touches)."""
import importlib
import pkgutil

import docagent


def test_all_modules_import():
    names = [m.name for m in pkgutil.walk_packages(docagent.__path__, "docagent.")]
    assert "docagent.evaluation" in names and "docagent.mcp_server" in names
    for name in names:
        importlib.import_module(name)
