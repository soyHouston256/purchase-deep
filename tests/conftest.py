import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from tests.fixtures import make_toy_env  # noqa: E402


@pytest.fixture
def env():
    cfg, cat, store, retriever, agent = make_toy_env()
    return {"cfg": cfg, "cat": cat, "store": store, "retriever": retriever, "agent": agent}
