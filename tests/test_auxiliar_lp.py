import numpy as np
import numpy.testing as npt
import pytest
from exceptions import UnfeasibleError
from pytest import input_test_data
import sys
import io

from auxiliar_lp import AuxiliarLP
from Utils.linear_algebra import LinearAlgebra


class TestAuxiliar:

    def test_auxiliar_lp_phase_1(self):
        """
        Test the first phase of simplex, auxiliar
        """
        baseTableau = np.array([
            [0, 0, 0, -3, -2, 0, 0, 0, 0],
            [1, 0, 0, 2, 1, 1, 0, 0, 8],
            [0, 1, 0, 1, 2, 0, 1, 0, 8],
            [0, 0, 1, 1, 1, 0, 0, 1, 5],
        ])

        pl = AuxiliarLP(baseTableau)

        result_tableau = pl.phase_1()

        expectedTableau = np.array(
            [[0., -1., 4., 0., 0., 0., -1., 4., 12],
             [0., -1., 2., 1., 0., 0., -1., 2., 2.],
             [1., 1., -3., 0., 0., 1., 1., -3., 1.],
             [0., 1., -1., 0., 1., 0., 1., -1., 3.]])

        npt.assert_allclose(result_tableau, expectedTableau)

    def test_auxiliar_tableau_pre_solve(self):
        """
        Test if C (tableau's first line) is correct after phase 1
        """
        baseTableau = np.array([
            [0, 0, 0, -3, -2, 0, 0, 0, 0],
            [1, 0, 0, 2, 1, 1, 0, 0, 8],
            [0, 1, 0, 1, 2, 0, 1, 0, 8],
            [0, 0, 1, 1, 1, 0, 0, 1, 5],
        ])

        pl = AuxiliarLP(baseTableau)
        presolved_tableau = pl.pre_solve_auxiliar_problem()

        resultC = presolved_tableau[0]

        # should represent a canonical form
        expectedC = [-1, -1, -1, -4, -4, -1, -1, -1, 0, 0, 0, -21]

        npt.assert_allclose(resultC, expectedC)

    def test_pre_solve_fixes_negative_b_entry(self):
        baseTableau = np.array([
            [0, 0, 0, -3, -2, 0, 0, 0, 0],
            [1, 0, 0, 2, 1, 1, 0, 0, -8],
            [0, 1, 0, 1, 2, 0, 1, 0, -8],
            [0, 0, 1, 1, 1, 0, 0, 1, -5],
        ])

        pl = AuxiliarLP(baseTableau)

        result_tableau = pl.pre_solve_auxiliar_problem()

        b_column = result_tableau[1:, -1]

        assert not LinearAlgebra.any_below_zero(b_column)

    def test_raises_unfeasible_exception_and_certificate(self):
        baseTableau = np.array([
            [0, 0, 0, -1, -1, -1, -1, 0],
            [1, 0, 0, 4, 10, -6, -2, 6],
            [0, 1, 0, -2, 2, -4, 1, 5],
            [0, 0, 1, -7, -2, 0, 4, 3],
        ])

        restrictions = LinearAlgebra.get_number_of_n_restrictions(baseTableau)

        pl = AuxiliarLP(baseTableau)

        with pytest.raises(UnfeasibleError) as exc:
            pl.phase_1()
        certificate = exc.value.certificate

        assert len(certificate) == restrictions, "Certificate should have the same size as the number of restrictions"

    def test_raises_unfeasible_exception(self):
        """
        When the lp is unfeasible, an exception should be raised with its certificate
        """
        baseTableau = np.array([
            [-2, 1, -2, 0],
            [-1, -2, 1, -1],
            [1, -1, 1, 3]
        ])

        simplexObj = AuxiliarLP(baseTableau)

        with pytest.raises(UnfeasibleError):
            simplexObj.phase_1()

    def test_raises_unfeasible_exception_with_certificate(self):
        baseTableau = np.array([
            [-6, -1, 1, 0],
            [5, 1, 1, 1],
            [-1, 1, 2, 5]
        ])

        auxiliar = AuxiliarLP(baseTableau)

        with pytest.raises(UnfeasibleError) as exc:
            auxiliar.phase_1()

        npt.assert_allclose(exc.value.certificate, [11, 1], err_msg="Certificate should be [11, 1]")

    def test_synthetic_restrictions_addition(self):
        baseTableau = np.array([
            [0, 0, 0, -3, -2, 0, 0, 0, 0],
            [1, 0, 0, 2, 1, 1, 0, 0, -8],
            [0, 1, 0, 1, 2, 0, 1, 0, 8],
            [0, 0, 1, 1, 1, 0, 0, 1, 5],
        ])

        pl = AuxiliarLP(baseTableau)

        # extract the ab matrix
        tableau = pl.pre_solve_auxiliar_problem()

        old_width = baseTableau.shape[1]

        assert tableau.shape[1] > old_width

    def test_canonical_form_creation_with_negative_b(self):
        baseTableau = np.array([[0, 0, 0, -3, -2, 0, 0, 0, 0],
                                [1, 0, 0, 2, 1, 1, 0, 0, 8],
                                [0, 1, 0, 1, 2, 0, 1, 0, 8],
                                [0, 0, 1, 1, 1, 0, 0, 1, -5]])

        pl = AuxiliarLP(baseTableau)

        tableau = pl.pre_solve_auxiliar_problem()

        expectedTableau = np.array([[-1., -1., 1., -2., -2., -1., -1., 1., 0., 0., 0., -21.],
                                    [1., 0., 0., 2., 1., 1., 0., 0., 1., 0., 0., 8.],
                                    [0., 1., 0., 1., 2., 0., 1., 0., 0., 1., 0., 8.],
                                    [0., 0., -1., -1., -1., 0., 0., -1., 0., 0., 1., 5.]])

        npt.assert_almost_equal(tableau, expectedTableau)

    def test_correct_c_restauration(self):
        """
        Test if the c vector is correctly restored after the phase 1, without putting in canonical form,
        as it must be equal to the original c vector (objective function)
        """
        base_tableau = np.array([
            [0, 0, 0, -2, -4, -8, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
        ])

        # creating the object and setting the old c value
        pl = AuxiliarLP(base_tableau)

        tableau_final = pl.phase_1(return_in_canonical=False)

        npt.assert_almost_equal(tableau_final[0], base_tableau[0])

    def test_auxiliar_lp_pre_solve_and_canonical_form_with_equal_nm(self):
        """
        Checks if we can pre solve the problem, making it ready for the simplex method run,
        which requires a trivial basis in canonical form
        """
        baseTableau = np.array([
            [0, 0, 0, -2, -4, -8, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
        ])

        aux = AuxiliarLP(baseTableau)

        tableau = aux.pre_solve_auxiliar_problem()

        expectedTableau = np.array([
            [-1., -1., -1., -1., -1., -1., -1., -1., -1., 0., 0., 0., -3.],
            [1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.],
            [0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1.],
            [0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1.]
        ])

        npt.assert_allclose(tableau, expectedTableau)

    def test_phase_1_simplex(self):
        baseTableau = np.array([
            [0, 0, 0, -2, -4, -8, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
        ])

        aux = AuxiliarLP(baseTableau)
        aux.phase_1()

        expectedOutput = np.array([
            [2, 4, 8, 0, 0, 0, 2, 4, 8, 14],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
        ])

        npt.assert_allclose(aux.tableau, expectedOutput)
