import numpy as np
import numpy.testing as npt
import pytest
from pytest import input_test_data
from Utils.linear_algebra import LinearAlgebra


class TestUtilLinearAlgebra:

    def test_solution_retrieval(self):
        tableau = [
            [1, 0, 1, 0, 0, 1, 0, 1, 13],
            [1, 0, -1, 1, 0, 1, 0, -1, 3],
            [1, 1, -3, 0, 0, 1, 1, -3, 1],
            [-1, 0, 2, 0, 1, -1, 0, 2, 2],
        ]

        calculated_solution = LinearAlgebra.get_solution(tableau)
        expected_solution = [3, 2, 0, 1, 0]
        npt.assert_allclose(calculated_solution, expected_solution)

    @pytest.mark.parametrize("entrada", input_test_data)
    def test_vero_removal(self, entrada):
        fullTableau = np.array(entrada.FullTableau)

        cleanedTableau = LinearAlgebra.drop_vero(fullTableau, entrada.N_restricoes)

        expected = entrada.Tableau

        npt.assert_allclose(cleanedTableau, expected)

    @pytest.mark.parametrize("entrada", input_test_data)
    def test_m_variables_calculation(self, entrada):
        tableau = np.array(entrada.FullTableau)

        calculated_m_variables = LinearAlgebra.get_number_of_m_variables(tableau, has_vero=True)

        expected = entrada.M_variaveis

        assert calculated_m_variables == expected

    @pytest.mark.parametrize("entrada", input_test_data)
    def test_m_variables_calculation_without_vero(self, entrada):
        tableau = np.array(entrada.Tableau)

        calculated_m_variables = LinearAlgebra.get_number_of_m_variables(tableau, has_vero=False)

        expected = entrada.M_variaveis

        assert calculated_m_variables == expected

    @pytest.mark.parametrize("entrada", input_test_data)
    def test_n_restrictions_calculation(self, entrada):
        tableau = np.array(entrada.Tableau)

        calculated_n_restrictions = LinearAlgebra.get_number_of_n_restrictions(tableau)

        expected = entrada.N_restricoes

        assert calculated_n_restrictions == expected

    def test_basic_feasible_column_extractor(self):
        input = np.array([
            [-1, -1, -1, -4, -4, -1, -1, -1, 0, 0, 0, -21],
            [1, 0, 0, 2, 1, 1, 0, 0, 1, 0, 0, 8],
            [0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 0, 8],
            [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 5],
        ])

        feasible_columns = LinearAlgebra.extract_feasible_columns(
            input, remove_b=True)

        expected = np.array([
            [-4, -4, -1, -1, -1, 0, 0, 0],
            [2, 1, 1, 0, 0, 1, 0, 0],
            [1, 2, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 0, 0, 1],
        ])
        npt.assert_allclose(feasible_columns, expected)

    def test_basic_column_retrieval(self):
        # adicionar mais casos
        sampleTableau = np.array([
            [0, 1, 1, 1, 0, 3, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # explicitly allow for b to be basic
        basicColumns = LinearAlgebra.findBasicColumns(sampleTableau, drop_vero=False, drop_b=False)

        expectedColumns = [4, 0, 6]

        npt.assert_allclose(basicColumns, expectedColumns)

    def test_advanced_column_retrieval_with_multiple_basis(self):
        """Should find leftmost basic column in case of multiple basic columns
        """
        sampleTableau = np.array([
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        ])

        basicColumns = LinearAlgebra.findBasicColumns(sampleTableau)

        expectedColumns = [7, 3, 9]

        npt.assert_allclose(basicColumns, expectedColumns)

    def test_advanced_column_retrieval_with_basic_b(self):
        # caso com vero e b
        sampleTableau = np.array([
            [0, 0, 0, 0, 1, 1, 1, 0, 3, 0, 0],
            [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        ])

        basicColumns = LinearAlgebra.findBasicColumns(sampleTableau)

        expectedColumns = [7, 3, 9]

        npt.assert_allclose(basicColumns, expectedColumns)

    def test_advanced_column_retrieval_with_non_canonic_tableau(self):
        # caso com vero e b, porem precisa droppar o c
        sampleTableau = np.array([
            [0, 0, 0, 0, 0, 1, 1, 1, 0],
            [1, 0, 0, 2, 1, 1, 0, 0, 8],
            [0, 1, 0, 1, 2, 0, 1, 0, 8],
            [0, 0, 1, 1, 1, 0, 0, 1, 5],
        ])

        basicColumns = LinearAlgebra.findBasicColumns(sampleTableau, drop_c=True)

        expectedColumns = [5, 6, 7]

        npt.assert_allclose(basicColumns, expectedColumns)

    def test_extract_feasible_columns(self):
        assert True
