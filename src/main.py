import numpy as np
import logging
from Utils.linear_algebra import LinearAlgebra

from tableau import TableauParsing
from simplex import Simplex
from auxiliar_lp import AuxiliarLP
from exceptions import UnfeasibleError, UnboundedError

logging.basicConfig(
    format='[%(filename)s:%(lineno)d] %(message)s',

)


class SimplexTester:
    """
    Atribuir ao longo da execução do programa e vou conferir se bate num teste de sistema.
    """

    def __init__(self):
        self.Tableau = None
        self.M_variables = None
        self.N_restrictions = None
        self.Auxiliary_Tableau = None
        self.Solved_Auxiliary_Tableau = None
        self.Tableau_With_First_Viable_Basis = None
        self.Final_Certificate = None
        self.Lp_Type = None


class SimplexRunner:
    def __init__(self) -> None:
        """ Cria o objeto e le a entrada
        """

        self.n_restrictions, self.m_variables = TableauParsing.read_n_m_dimensions()
        self.tableau = TableauParsing.read_ab_and_create_tableau(self.n_restrictions, self.m_variables)

        self.simplex = None

    def print_certificate(self):
        firstRow = self.tableau[0]
        certifate = firstRow[:self.n_restrictions]
        LinearAlgebra.arrayPrint(certifate)

    def print_x_solution(self):
        x_solution = LinearAlgebra.get_solution(self.tableau)
        x_solutions_without_aux_variables = x_solution[:self.m_variables]
        LinearAlgebra.arrayPrint(x_solutions_without_aux_variables)

    def get_optimal_value(self):
        """Gets value at last column of first row, i.e, optimal value

        Returns:
            _type_: _description_
        """
        return self.tableau[0][-1]

    def runSimplex(self):
        try:
            # execute phase 1

            auxiliar = AuxiliarLP(self.tableau, self.m_variables, self.n_restrictions)
            tableau_with_trivial_basis = auxiliar.phase_1()

            # execute phase 2
            phase2 = Simplex(m=self.m_variables, n=self.n_restrictions, tableau=tableau_with_trivial_basis)
            phase2.solve()

            self.tableau = phase2.tableau
            print("otima")
            print(self.get_optimal_value())
            self.print_x_solution()
            self.print_certificate()
            """
            otima
            36.0000000
            0.0000000 3.6000000 
            0.0000000 2.0000000 0.0000000 0.0000000 
            """

        # finish until done or unbounded

        except UnboundedError:
            print("ilimitada")
            print(self.print_certificate())

            pass
        except UnfeasibleError:
            print("inviavel")
            print(self.print_certificate())

            pass


# run main
if __name__ == "__main__":
    tmp = SimplexRunner()
    tmp.runSimplex()
