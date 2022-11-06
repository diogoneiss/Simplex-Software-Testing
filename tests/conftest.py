import pytest
import os
from pathlib import Path
from pytest_cases import fixture

import sys

ADD_TO_SYSPATH = True

# I dont know if this is needed
# tried and had no effect
if ADD_TO_SYSPATH:
    print("\nEntering conftest.py (first file before tests run) ")
    PROJECT_DIR = Path(__file__).resolve().parents[1] / "src"
    print("Project dir: ", PROJECT_DIR)

    sys.path.append(str(PROJECT_DIR))
    sys.path.append(str(PROJECT_DIR / "Utils"))
    print("sys.path: ")
    print(sys.path)
    print("____")

# delayed import to garantee that the sys.path is updated
from Utils.read_json import inject_test_data


@fixture(scope="module")
def input_data():
    return get_json_input()


def pytest_configure():
    pytest.input_test_data = get_json_input()


def get_json_input() -> list:
    """ Retorna a lista dentro do arquivo cases/input.json
    """
    working_directory = os.path.dirname(__file__)
    test_data = inject_test_data(file=Path("cases/input.json"), currentDir=working_directory)
    test_data = test_data.rawInputs
    return test_data
