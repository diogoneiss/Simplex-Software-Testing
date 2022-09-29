import pytest
import os
from pathlib import Path
from Utils.read_json import inject_test_data


def pytest_configure():
    pytest.input_test_data = get_json_input()


def get_json_input() -> list:
    """ Retorna a lista dentro do arquivo cases/input.json
    """
    working_directory = os.path.dirname(__file__)
    test_data = inject_test_data(file=Path("cases/input.json"), currentDir=working_directory)
    test_data = test_data.rawInputs
    return test_data
