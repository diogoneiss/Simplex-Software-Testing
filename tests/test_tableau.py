import numpy as np
import numpy.testing as npt
import pytest
from pytest import input_test_data
import sys
import io
from .conftest import input_data
from tableau import TableauParsing
from src.Utils.linear_algebra import LinearAlgebra
from pytest_cases import parametrize, fixture_ref


class TestTableau:
    @pytest.mark.parametrize("entrada", input_test_data)
    def test_read_dimensions(self, entrada):
        sys.stdin = io.StringIO(entrada.input)
        n_restrictions, m = TableauParsing.read_n_m_dimensions()

        assert m == entrada.M_variaveis
        assert n_restrictions == entrada.N_restricoes

    def test_read_input(self):
        assert True

    def test_read_and_create_tableau(self):
        assert True

    def test_tableau_slack_variables_with_3_variables(self):
        ab = np.array([[1, 2, 3, 50],
                       [4, 5, 6, 50],
                       [7, 8, 9, 50],
                       [10, 11, 12, 50]])
        n_rows = 4
        m_columns = 3
        c = np.array([101, 102, 103])

        tableau = TableauParsing.create_full_tableau(c, ab, n_rows, m_columns)

        # last n_rows+1 should be an identity and B, as they are the slack variables
        # Tableau has form, with b as single column
        # [ 0    | C | 0 | 0 ] (1st row)
        # [ Vero | A | I | b ] (n-1 rows)
        # so I need to get just the last n + 1 columns

        # begin slice with n_rows + 1 from the end
        offset = n_rows + 1

        # slack_related_sections = tableau.T[-offset:]
        slack_related_sections = tableau[:, -offset:]

        # Assert - Creating the desired format

        b_values = [0, 50, 50, 50, 50]
        # convert b
        slack_b_expected = np.reshape(b_values, (-1, 1))
        slack_a_expected = np.identity(n_rows)

        # i don't need the additional 0, just n_rows, as the extra zero is in the first row of b (which is really w)
        slack_c_expected = [0, 0, 0, 0]

        # merge ab and then c and ab
        slack_ac_expected = np.vstack((slack_c_expected, slack_a_expected))
        slack_c_ab_expected = np.hstack((slack_ac_expected, slack_b_expected))

        npt.assert_allclose(slack_related_sections, slack_c_ab_expected)

    def test_tableau_first_line(self):
        """Método que vai testar se o C é corretamente montado de acordo com a largura de A
        """
        array_c = np.array([1, 2, 3])
        n_rows = 4
        m_columns = 3

        random_a = np.arange((m_columns + 1) * n_rows, ).reshape((4, -1))

        result_tableau = TableauParsing.create_full_tableau(array_c, random_a, n_rows, m_columns)

        first_line = result_tableau[0]

        # deve ser igual a | -c + 0n |
        expected_line = np.array([0, 0, 0, 0, -1, -2, -3, 0, 0, 0, 0, 0])

        npt.assert_allclose(first_line, expected_line)

    @pytest.mark.parametrize("sample_input", input_test_data)
    def test_operations_register_existence(self, sample_input):
        # testar se o registro de operações é corretamente montado
        """ cria uma matriz com a primeira linha sendo de 0's e o restante sendo uma identidade
        para constituir o VERO. Formato:
        | 0 .... 0 0 0 |
        | 1 .... 0 0 0 |
        | 0 .... 1 0 0 |
        | 0 .... 0 1 0 |
        | 0 .... 0 0 1 |
        """
        m = sample_input.M_variaveis
        n = sample_input.N_restricoes
        c = sample_input.C
        ab = np.array(sample_input.AB)

        created_tableau = TableauParsing.create_full_tableau(c, ab, n, m)

        # get all rows x first n columns
        extracted_vero = created_tableau[:, :n]

        c_fill = np.zeros(n)
        ab_fill = np.identity(n)
        expected_register = np.vstack((c_fill, ab_fill))

        npt.assert_allclose(extracted_vero, expected_register)

    # fazer isso tudo mockando o input com o Typer

    @pytest.mark.parametrize("entrada", input_test_data)
    def test_input_read(self, entrada):
        sys.stdin = io.StringIO(entrada.input)

        _, _, tableau = TableauParsing.read_everything_and_create_tableau()

        expected_tableau = entrada.FullTableau

        npt.assert_allclose(tableau, expected_tableau)

    #    @parametrize("entrada", [fixture_ref(*input_data)])
    @parametrize("entrada", input_test_data)
    def test_full_input_read(self, entrada):
        sys.stdin = io.StringIO(entrada.input)

        n_rows, m_columns, _ = TableauParsing.read_everything_and_create_tableau()

        n_esperado = entrada.N_restricoes
        m_esperado = entrada.M_variaveis

        assert n_esperado, n_rows
        assert m_columns, m_esperado
