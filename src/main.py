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
        self.tableau, self.n_restrictions = TableauParsing.read_ab_and_create_tableau(self.n_restrictions,
                                                                                      self.m_variables)

        self.simplex = None

    def print_certificate(self, certificate=None):
        if certificate is None:
            certificate = LinearAlgebra.retrive_certificate(self.tableau, self.n_restrictions)
        LinearAlgebra.arrayPrint(certificate)

    def print_x_solution(self):
        x_solution = LinearAlgebra.get_solution(self.tableau)
        x_solutions_without_aux_variables = x_solution[:self.m_variables]
        LinearAlgebra.arrayPrint(x_solutions_without_aux_variables)

    def get_optimal_value(self):
        return self.tableau[0][-1]

    def run_simplex(self):
        try:
            # execute phase 1
            if not self.__should_skip_auxiliar():
                # if there is a trivial solution, skip auxiliar
                tableau_with_trivial_basis = AuxiliarLP(self.tableau).phase_1()
            else:
                tableau_with_trivial_basis = self.tableau

            # execute phase 2
            phase2 = Simplex(m=self.m_variables, n=self.n_restrictions, tableau=tableau_with_trivial_basis)

            phase2.solve()

            self.tableau = phase2.tableau
            print("otima")
            print(self.get_optimal_value())
            self.print_x_solution()
            self.print_certificate()

        except UnboundedError as Ub:
            print("ilimitada")
            self.print_x_solution()
            self.print_certificate(Ub.certificate)
        except UnfeasibleError as Uf:
            print("inviavel")
            self.print_certificate(Uf.certificate)

    def __should_skip_auxiliar(self):
        basic_columns = LinearAlgebra.findBasicColumns(self.tableau)

        # if there is a trivial solution, skip auxiliar
        trivial_basis_found = np.all(basic_columns != -1)

        b_column = self.tableau.T[-1]
        # if there is a negative b value the trivial solution is unfeasible
        trivial_solution_is_feasible = not LinearAlgebra.any_below_zero(b_column)

        return trivial_solution_is_feasible and trivial_basis_found


# run main
if __name__ == "__main__":
    tmp = SimplexRunner()
    tmp.run_simplex()
