import math

import numpy as np
import logging
from Utils.linear_algebra import LinearAlgebra
from exceptions import UnboundedError, UnfeasibleError


class Simplex:

    def __init__(self, m, n, tableau) -> None:

        self.m_variables = m
        self.n_restrictions = n
        # if tableau is a list, convert to np.ndarray
        if isinstance(tableau, list):
            self.tableau = np.array(tableau, dtype=float)
        else:
            self.tableau = tableau.astype(float)

    def solve(self):

        self.__remove_values_lower_than_tolerance()

        # sanity check, there should be no negative b values(last column)
        b_column = self.tableau[1:, -1]

        if LinearAlgebra.any_below_zero(b_column):
            raise Exception(f"Negative b value inputed at b column {b_column}")

        self.assert_not_unbounded()

        stop = self.isSimplexDone()

        while not stop:

            # pivot
            row, column = self.findPivot(self.tableau, n_restrictions=self.n_restrictions)

            # this happens when an unfeasible problem is found
            if column == -1 or row == -1:
                break
                # obj_value = self.tableau[0][-1]
                #
                # # This is not needed, as we can create rare problemns where the objective function is < 0
                # if LinearAlgebra.smaller_than_zero(obj_value):
                #     certificate = LinearAlgebra.retrive_certificate(self.tableau, self.n_restrictions)
                #     raise UnfeasibleError(certificate)

            self.tableau = self.pivotTableau(self.tableau, row=row, column=column)
            self.__remove_values_lower_than_tolerance()

            stop = self.isSimplexDone()

            # check if it became unbounded
            self.assert_not_unbounded()

        return self.tableau

    def __remove_values_lower_than_tolerance(self):
        self.tableau = LinearAlgebra.replace_values_smaller_then_tol(self.tableau)

    def assert_not_unbounded(self):
        is_unbounded = self.isUnbounded(self.tableau)
        if is_unbounded:
            # print(f"Unbounded problem found at column {column}")
            certificate = LinearAlgebra.retrive_certificate(self.tableau, self.n_restrictions)
            raise UnboundedError(certificate)

    @staticmethod
    def isUnbounded(tableau: np.ndarray):

        for column in tableau.T:
            if column[0] < 0:
                is_negative_array = column <= 0

                if all(is_negative_array):
                    return True

        return False

        # for column in tableau[:, ]:
        #
        #     # vetor booleano se a coluna é menor que zero
        #     isNegativeArr = np.where(column < 0)
        #
        #     # lembrar c < 0 no tableau implica em c > 0 na funcao objetivo, pq aqui ele está multiplicado
        #     # por -1, entao preciso ver se ele esta positivo no tableau
        #     c_negative_in_objective_function = not isNegativeArr[0]
        #     # se para um x_i seu c verdadeiro é < 0 e o Ai é completamente
        #     # negativo, podemos concluir que podemos aumentar esse x_i infinitamente, junto de aumentar outro
        #     # x_j "normal", sem violar nenhuma restricao
        #     allNegative = np.all(isNegativeArr[1:])
        #
        #     if allNegative and c_negative_in_objective_function:
        #         return True
        #
        # return False

    def isSimplexDone(self):
        """
        Checks if the simplex is done, that is, not more pivotable columns
        :return:
        """

        start = self.n_restrictions
        end = -1
        c_slice = self.tableau[0][start:end]

        if LinearAlgebra.any_below_zero(c_slice):
            return False

        # retirei pro caso de termos
        # for b in self.tableau[:, -1]:
        #    if b < 0:
        #        return False

        return True

    @staticmethod
    def putInCanonicalForm(original_tableau: np.ndarray):
        # for each basic column, subtract the column from the objective function c times such
        # that the basic column is zero in the first row

        basic_columns = LinearAlgebra.findBasicColumns(original_tableau, drop_c=True)

        pivoted_tableau = original_tableau
        # print(basic_columns)
        # matprint(pivoted_tableau)
        for restriction, x_index in enumerate(basic_columns):

            # pivoting at the c_i, such that i is the ith variable in basis
            c_index = pivoted_tableau[0, x_index]
            if c_index == 0:
                # print(f"C pivot with value {c_index} at i = {x_index}, skipping")
                continue

            # print(f"C pivot with value {c_index} at i = {x_index}")
            pivoted_tableau = Simplex.pivotTableau(pivoted_tableau, column=x_index, row=restriction + 1)

            # the first row is the objective function
            # and the i_th item is the current pivot
            # c_i = original_tableau[0, x_index]

            # variable_row = np.where(original_tableau[1:, basicColumn] == 1)[0][0] + 1

            # subtract the basis (an identity) times c_i, resulting in c_i = 0
            # original_tableau[0] -= c_i * original_tableau[variable_row]
        # print("___")
        # matprint(pivoted_tableau)
        return pivoted_tableau

    @staticmethod
    def findPivot(original_tableau: np.ndarray, n_restrictions: int):
        # find column < 0
        column_i = -1

        feasible_c_columns = LinearAlgebra.extract_feasible_columns(original_tableau, n_restrictions)

        # use bland rule (leftmost c value)
        for i, value in enumerate(feasible_c_columns[0]):
            if value < 0:
                # need to re-add the vero
                column_i = i + n_restrictions
                break

        row = -1
        smallestRatio = np.inf

        if column_i == -1:
            return -1, -1

        # find smallest ratio (b_i/a_i), such that a_i > 0

        for j, a_j in enumerate(original_tableau.T[column_i]):
            b_j = original_tableau[j][-1]
            if a_j > 0:
                ratio = b_j / a_j

                if ratio < smallestRatio:
                    smallestRatio = ratio
                    row = j

        # if column_i == -1:
        #     certificate = LinearAlgebra.retrive_certificate(original_tableau, n_restrictions)
        #     raise UnfeasibleError(certificate)

        return row, column_i

    @staticmethod
    def pivotTableau(original_tableau: np.ndarray, column: int, row: int):

        # se passar lista inves de np.array() não quebra
        if type(original_tableau) is list:
            tableau = np.array(original_tableau, dtype=float)
        else:
            tableau = np.copy(original_tableau)

        num_rows, _ = tableau.shape

        pivotableRows = list(range(num_rows))

        pivotValue = tableau[row][column]
        # print(f"Pivoting {pivotValue} at [{row},{column}] ")

        if pivotValue == 0:
            logging.fatal(f"Pivoting by 0 at tableau[{row}, {column}]")

        # make pivot 1
        tableau[row] = tableau[row] * (1.0 / pivotValue)

        # remove pivot row from list, as it is already done
        idxPivot = pivotableRows.index(row)
        pivotableRows.pop(idxPivot)

        # print("Pivotable rows", pivotableRows)

        # make each value in column zero
        for current_row in pivotableRows:
            """
            Se eu tenho
            [[3 2 3],
            [1 4 5]]
            e quero transformar o 3 em 0 , preciso aplicar qual operacao na coluna?

            Subtrair 3 * linhaPivo (o pivo já vai ser 1), ou seja:

            [3, 2, 3] = [3, 2, 3] - [1, 4, 5] * 3
            == [0, - 10, -12]

            """

            b_i = tableau[current_row, -1]

            # TODO: Ver se isso é a melhor maneira de lidar com b negativo
            # TODO: Criar um método numericamente seguro de comparação com zero (tolerancia de proximidade)
            # if b_i < 0:
            #    tableau[current_row] = tableau[current_row] * - 1

            # if the current_row is 0, this will do nothing
            rowSubtractor = tableau[row] * tableau[current_row][column]
            # print("I want to zero ")
            # print(f"Subtracting {rowSubtractor} from {tableau[current_row]} ")
            tableau[current_row] -= rowSubtractor

        return tableau
