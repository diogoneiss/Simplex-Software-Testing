import numpy as np
import numpy.testing as npt
import pytest
from pytest import input_test_data

from tableau import TableauParsing


class TestTableau:
    def test_read_dimensions(self):
        assert True

    def test_read_input(self):
        assert True

    def test_read_and_create_tableau(self):
        assert True

    def test_tableau_slack_variables_with_3_variables(self):
        # adicionar mais casos
        arrayAB = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        n = 4
        m = 3
        tableauBase = TableauParsing.add_auxilliary_variables_to_a(arrayAB, n, m)
        # deve ser igual ao arrayAB, exceto com a identidade ao final
        expectedTableau = np.array([[1, 2, 3, 1, 0, 0, 0],
                                    [4, 5, 6, 0, 1, 0, 0],
                                    [7, 8, 9, 0, 0, 1, 0],
                                    [10, 11, 12, 0, 0, 0, 1]])

        npt.assert_allclose(tableauBase, expectedTableau)

    def test_tableau_first_line(self):
        """Método que vai testar se o C é corretamente montado de acordo com a largura de A
        """
        arrayC = [1, 2, 3]
        n = 4
        m = 3
        first_line = TableauParsing.mount_first_line(arrayC, n)

        # deve ser igual a | -c + 0n |
        expected_line = np.array([-1, -2, -3, 0, 0, 0, 0, 0])

        npt.assert_allclose(first_line, expected_line)

    @pytest.mark.parametrize("input", input_test_data)
    def test_operations_register(self, input):
        # testar se o registro de operações é corretamente montado
        """ cria uma matriz com a primeira linha sendo de 0's e o restante sendo uma identidade
        para constituir o VERO. Formato:
        | 0 .... 0 0 0 |
        | 1 .... 0 0 0 |
        | 0 .... 1 0 0 |
        | 0 .... 0 1 0 |
        | 0 .... 0 0 1 |
        """
        n = input.N_restricoes

        combined = TableauParsing.create_operations_register(n)

        expected_register = input.VERO

        npt.assert_allclose(combined, expected_register)
