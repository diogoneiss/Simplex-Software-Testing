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

"""
Para entendimento, temos um arquivo main, que possui a classe SimplexRunner, que executa a fase 1 e 2

* auxiliar_lp.py: responsavel pelo auxiliar
* exceptions.py, que define as exceções utilizadas no programa no caso de inviavel ou ilimitada
* tableau.py, que lê o arquivo de entrada e cria o tableau no formato correto

Dentro da pasta Utils temos o arquivo linear_algebra.py, que possui funções úteis e modulares para lidar com vários aspectos do simplex.

Esse trabalho foi feito para a disciplina de teste de software, então temos uma pasta /tests, com vários testes em desenvolvimento dentro.

"""
class SimplexTester:
    """
    Atribuir ao longo da execução do programa e vou conferir se bate num teste de sistema.
    Eu uso isso para a entrega de teste de software, nao de pesquisa operacional
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

        # We can have a smaller n, if we have dependent restrictions
        original_n, self.m_variables = TableauParsing.read_n_m_dimensions()
        self.tableau, self.n_restrictions = TableauParsing.read_ab_and_create_tableau(original_n,
                                                                                      self.m_variables)

        self.original_n = original_n

    def print_certificate(self, certificate=None):
        if certificate is None:
            certificate = LinearAlgebra.retrive_certificate(self.tableau, self.n_restrictions)

        # add missing 0's to certificate, corresponding to the removed restrictions
        # happens when we have dependent restrictions, ie, one scaled by a constant
        if self.original_n > self.n_restrictions:
            delta_zeros = self.original_n - self.n_restrictions
            certificate = np.append(certificate, np.zeros(delta_zeros))

        LinearAlgebra.arrayPrint(certificate)

    def print_x_solution(self):

        x_solution = LinearAlgebra.get_x_solution(self.tableau)
        LinearAlgebra.arrayPrint(x_solution)

    def get_optimal_value(self):
        optimal = self.tableau[0][-1]
        return round(optimal, 7)

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
            LinearAlgebra.arrayPrint(Ub.x_solution)
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


def main():
    simplex_runner = SimplexRunner()
    simplex_runner.run_simplex()


# run main
if __name__ == "__main__":
    main()
