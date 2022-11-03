import logging
import math

import numpy as np

from simplex import Simplex
from Utils.linear_algebra import LinearAlgebra
from exceptions import UnfeasibleError


class AuxiliarLP:
    def __init__(self, tableau: np.ndarray):

        # sanity check
        if isinstance(tableau, list):
            logging.warning("List passed to auxiliar_lp instead of numpy array", tableau)
            tableau = np.array(tableau)

        self.tableau = tableau

        self.m_variables = LinearAlgebra.get_number_of_m_variables(tableau)
        self.n_restrictions = LinearAlgebra.get_number_of_n_restrictions(tableau)

        # synthetic variables count
        self.slack_variables_added = 0

        # synthetic columns
        self.auxiliary_columns = []

        self.old_c = tableau[0]

    def __run_auxiliar_lp(self):

        # put sythetic columns in canonical form
        canonical_tableau = self.setup_auxiliar_problem()

        # create simplex object with the new tableau and variables
        n = LinearAlgebra.get_number_of_n_restrictions(canonical_tableau)
        m = LinearAlgebra.get_number_of_m_variables(canonical_tableau)

        # run simplex
        self.tableau = Simplex(m=m, n=n, tableau=canonical_tableau).solve(phase1=True)

        print(f"Objective value for auxiliar: {self.tableau[0][-1]}")

        # if a 0 value objective function is not found then it is unfeasible
        if self.is_unfeasible():
            certificate = LinearAlgebra.retrive_certificate(self.tableau, self.n_restrictions)
            raise UnfeasibleError(certificate)

    def phase_1(self):

        # run simplex to try to get to zero value objective function
        # throws exception if unfeasible
        self.__run_auxiliar_lp()

        new_c = self.tableau[0]
        old_c = self.old_c

        SIMULATE_AUXILIAR_OPERATIONS_IN_C = False

        # remove synthetic columns (2*n+m to 2*n+m+k, such that k is the new columns count ) and restore c
        start_synthetic = self.m_variables + 2 * self.n_restrictions
        self.tableau = np.delete(self.tableau, np.s_[start_synthetic: start_synthetic + self.slack_variables_added],
                                 axis=1)

        if SIMULATE_AUXILIAR_OPERATIONS_IN_C:
            self.__restore_original_c()
        else:
            self.tableau[0] = self.old_c

        # after c is reinserted we need to redo the canonical form (make the basis columns canonical)
        self.tableau = Simplex.putInCanonicalForm(self.tableau)

        return self.tableau

    def __change_objective_function(self):
        # tableau tem m (vero) + n (variaveis) + m (folgas, opcionais) + 1 de largura

        # here we have vero, variables, slack, new variables and b. They all should be zero
        new_c = np.zeros(self.tableau.shape[1] + self.slack_variables_added)

        # make the obj value for auxiliary columns 1
        for c_i in self.auxiliary_columns:
            new_c[c_i] = 1

        return new_c

    def __merge_new_c_and_new_restrictions(self, new_ab, new_c):
        full_tableau = np.vstack((new_c, new_ab))

        return full_tableau

    def __add_new_synthetic_restrictions(self):

        # remove first row from tableau
        abMatrix = np.delete(self.tableau, 0, 0)

        if self.slack_variables_added > 0:
            # create n_restrictions * new_m_variables identity
            identityRestrictions = np.zeros((self.n_restrictions, self.slack_variables_added))

            # index of last column, before b
            offset = self.tableau.shape[1] - 1

            # insert in basis
            for i, index in enumerate(self.auxiliary_columns):
                identityRestrictions[index - offset][i] = 1

            abMatrix = np.insert(abMatrix, offset, identityRestrictions, axis=1)

        return abMatrix

    def setup_auxiliar_problem(self):

        # check to see if we need to insert synthetic variables, will always do nothing in Ax<=b problem
        self.__change_to_auxiliary_problem()

        # use the linear algebra find basis method
        start = self.m_variables + 2 * self.n_restrictions
        new_tableau = self.tableau.copy()

        # pivotear cada coluna da base auxiliar para colocar o c_i 0 zero
        for i, col in enumerate(self.auxiliary_columns):
            variable_index = i + 1
            new_tableau = Simplex.pivotTableau(new_tableau, col, variable_index)

        return new_tableau

    def __change_to_auxiliary_problem(self):
        slack_start = self.m_variables + self.n_restrictions

        # if the tableau has additional columns with n_restrictions width, we have a full identity added
        # this should ALWAYS be true
        if self.tableau.shape[1] - 1 == slack_start + self.n_restrictions:
            print("Full identity found")
            self.auxiliary_columns = list(range(slack_start, slack_start + self.n_restrictions))
            self.slack_variables_added = 0
        else:
            # check missing variables
            # trivial_basis = LinearAlgebra.findBasicColumns(self.tableau, drop_c=True, get_rightmost=True)
            # I will not do this as it will never be needed, but if needed
            # this behaviour can be achived by checking which variables do not have a
            # slack added (index of the basis is below the slack start
            # and add then, remebering to flag this addition in the object properties
            raise NotImplemented("We do not have aux variable developed")

        new_c = self.__change_objective_function()
        new_ab = self.__add_new_synthetic_restrictions()
        self.old_c = self.tableau.T[0]
        self.tableau = self.__merge_new_c_and_new_restrictions(new_ab, new_c)

    def __is_synthetic_variable(self, index):
        return index in self.auxiliary_columns

    def is_unfeasible(self):

        result = self.tableau[0][-1]
        print("Unfeasible test")
        LinearAlgebra.matprint(self.tableau)
        # se o resultado for 0, é otimo.
        # TODO: Ver caso do livro do Thie, que o Scipy resolve
        if math.isclose(result, 0):
            basic_variables = LinearAlgebra.findBasicColumns(self.tableau, self.n_restrictions, True)
            logging.debug("basic_variables", basic_variables)

            for i, x_index in enumerate(basic_variables):
                # means that the variable in the basis is a synthetic variable
                if self.__is_synthetic_variable(x_index):
                    print(f"x index {x_index} is synthetic, ie, greater or equal than {self.m_variables}")
                    LinearAlgebra.matprint(self.tableau)
                    return True

            return False

        # se o resultado for negativo, é inviavel
        if result < 0:
            return True

    def __restore_original_c(self):
        originalC = self.old_c

        # get only first n values from row
        veroData = self.tableau[0][0:self.n_restrictions]

        # perform operations to get the new c
        for i, value in enumerate(veroData):
            # i + 1 is the 0th restriction row
            operationRow = self.tableau[i + 1]
            # print("operationRow", operationRow)
            originalC += (value * operationRow)

        self.tableau[0] = originalC
