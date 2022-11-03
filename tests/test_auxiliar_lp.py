import numpy as np
import numpy.testing as npt
import pytest
from pytest import input_test_data
import sys
import io

from auxiliar_lp import AuxiliarLP
from Utils.linear_algebra import LinearAlgebra


class TestAuxiliar:

    def test_auxiliar_lp_simplex(self):
        baseTableau = np.array([
            [0, 0, 0, -3, -2, 0, 0, 0, 0],
            [1, 0, 0, 2, 1, 1, 0, 0, 8],
            [0, 1, 0, 1, 2, 0, 1, 0, 8],
            [0, 0, 1, 1, 1, 0, 0, 1, 5],
        ])
        m_variaveis = 2
        n_restricoes = 3

        pl = AuxiliarLP(baseTableau)

        result_tableau = pl.phase_1()


        LinearAlgebra.matprint(result_tableau)

        expectedTableau = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
             [0., -1., 2., 1., 0., 0., -1., 2., 0., -1., 2., 2.],
             [1., 1., -3., 0., 0., 1., 1., -3., 1., 1., -3., 1.],
             [0., 1., -1., 0., 1., 0., 1., -1., 0., 1., -1., 3.]])

        npt.assert_allclose(result_tableau, expectedTableau)

    def test_auxiliar_tableau(self):
        baseTableau = np.array([
            [0, 0, 0, -3, -2, 0, 0, 0, 0],
            [1, 0, 0, 2, 1, 1, 0, 0, 8],
            [0, 1, 0, 1, 2, 0, 1, 0, 8],
            [0, 0, 1, 1, 1, 0, 0, 1, 5],
        ])

        pl = AuxiliarLP(baseTableau)
        pl.setup_auxiliar_problem()

        resultC = pl.tableau[0]

        # largura 2m + n + 1
        expectedC = [0, 0, 0, 0, 0, 1, 1, 1, 0]

        npt.assert_allclose(resultC, expectedC)

    def test_SyntheticRestrictionAddition(self):
        baseTableau = np.array([
            [0, 0, 0, -3, -2, 0, 0, 0, 0],
            [1, 0, 0, 2, 1, 1, 0, 0, 8],
            [0, 1, 0, 1, 2, 0, 1, 0, 8],
            [0, 0, 1, 1, 1, 0, 0, 1, 5],
        ])

        m_variaveis = 2
        n_restricoes = 3

        pl = AuxiliarLP(baseTableau)

        cArray = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]

        tableau = pl._AuxiliarLP__add_synthetic_restrictions(cArray)

        expectedTableau = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                           [1, 0, 0, 2, 1, 1, 0, 0, 1, 0, 0, 8],
                           [0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 0, 8],
                           [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 5]]

        npt.assert_almost_equal(tableau, expectedTableau)

    def test_canonical_form_creation(self):
        baseTableau = np.array([[0, 0, 0, -3, -2, 0, 0, 0, 0],
                                [1, 0, 0, 2, 1, 1, 0, 0,
                                 8], [0, 1, 0, 1, 2, 0, 1, 0, 8],
                                [0, 0, 1, 1, 1, 0, 0, 1, 5]])

        m_variaveis = 2
        n_restricoes = 3

        pl = AuxiliarLP(baseTableau)

        tableau = pl.setup_auxiliar_problem()

        LinearAlgebra.matprint(tableau)

        expectedTableau = [
            [-1, -1, -1, -4, -4, 0, 0, 0, -21],
            [1, 0, 0, 2, 1, 1, 0, 0, 8],
            [0, 1, 0, 1, 2, 0, 1, 0, 8],
            [0, 0, 1, 1, 1, 0, 0, 1, 5],
        ]

        npt.assert_almost_equal(tableau, expectedTableau)

    def test_apply_cumulative_vero_operations(self):
        baseTableau = np.array([
            [-1, -1, -1, -4, -4, 0, 0, 0, -21],
            [1, 0, 0, 2, 1, 1, 0, 0, 8],
            [0, 1, 0, 1, 2, 0, 1, 0, 8],
            [0, 0, 1, 1, 1, 0, 0, 1, 5],
        ])

        m_variaveis = 2
        n_restricoes = 3

        ## need to set up the old_c manually, as I'm passing a pivoted tableau
        old_c = [0, 0, 0, -3, -2, 0, 0, 0, 0]
        pl = AuxiliarLP(baseTableau)
        pl.old_c = np.array(old_c)

        tableau_final = pl.phase_1()

        calculatedC = tableau_final[0]
        expectedTableau = [-1, -1, -1, -7, -6, -1, -1, -1, -21]

        npt.assert_almost_equal(calculatedC, expectedTableau)

    def test_auxiliar_lp_canonical_form(self):
        baseTableau = np.array([
            [0, 0, 0, -2, -4, -8, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
        ])

        aux = AuxiliarLP(baseTableau)

        tableau = aux.setup_auxiliar_problem()

        expectedTableau = np.array([
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
        ])

        npt.assert_allclose(tableau, expectedTableau)

    def test_phase_1(self):
        baseTableau = np.array([
            [0, 0, 0, -2, -4, -8, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
        ])

        aux = AuxiliarLP(baseTableau)
        aux.phase_1()

        expectedOutput = np.array([
            [2, 4, 8, 2, 4, 8, 14],
            [1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1]
        ])

        npt.assert_allclose(aux.tableau, expectedOutput)
