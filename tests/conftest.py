# conftest.py
import pytest
import multiprocessing as mp

@pytest.fixture(autouse=True)
def set_spawn_method():
    mp.set_start_method('spawn', force=True)