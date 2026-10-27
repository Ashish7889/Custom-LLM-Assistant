"""
tests/conftest.py -- Pytest configuration and environment patches.
"""

import sys

# Prevent Windows Python 3.13 pyarrow Cython access-violation crash during dynamic imports
sys.modules["pyarrow"] = None
