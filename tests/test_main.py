import pytest
import numpy
import numpy.testing as npt
import sys
import io

from main import SimplexRunner


class TestSimplexRunner:

    @pytest.fixture(scope="class")
    def entrada(self):
        value = "4 2\n-2 10\n-5 -1 6\n10 5 18\n19 0 2\n5 -3 0"
        return value

    def test_full_read(self, entrada):
        sys.stdin = io.StringIO(entrada)

        obj = SimplexRunner()

        m = 2
        n = 4
        """

        c = [-2, 10]
        ab = 
            -5 -1 6
            10 5 18
            19 0 2
            5 -3 0
        """
        final_tableau = [
            [0, 0, 0, 0, 2, -10, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, -5, -1, 1, 0, 0, 0, 6],
            [0, 1, 0, 0, 10, 5, 0, 1, 0, 0, 18],
            [0, 0, 1, 0, 19, 0, 0, 0, 1, 0, 2],
            [0, 0, 0, 1, 5, -3, 0, 0, 0, 1, 0],

        ]

        assert obj.m_variables == m
        assert obj.n_restrictions == n

        npt.assert_allclose(obj.tableau, final_tableau)

    def test_print_certificate(self):
        assert True

    def test_print_x_solution(self):
        assert True

    def test_get_optimal_value(self):
        entrada = "3 3\n2 4 8 \n1 0 0 1\n0 1 0 1\n0 0 1 1"
        sys.stdin = io.StringIO(entrada)

        obj = SimplexRunner()

        obj.runSimplex()

        optimal_value = obj.get_optimal_value()

        npt.assert_allclose(optimal_value, 14)

    def test_run_simplex(self):
        assert True
