import logging
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
        canonical_tableau = self.pre_solve_auxiliar_problem()

        # create simplex object with the new tableau and variables
        n = LinearAlgebra.get_number_of_n_restrictions(canonical_tableau)
        m = LinearAlgebra.get_number_of_m_variables(canonical_tableau)

        # run simplex
        runner = Simplex(m=m, n=n, tableau=canonical_tableau)
        self.tableau = runner.solve()

        # if a 0 value objective function is not found then it is unfeasible
        if self.is_unfeasible():
            certificate = LinearAlgebra.retrive_certificate(self.tableau, self.n_restrictions)
            raise UnfeasibleError(certificate)

    def phase_1(self, return_in_canonical=True, simulate_auxiliar_operations_in_c=False):

        # run simplex to try to get to zero value objective function
        # throws exception if unfeasible
        self.__run_auxiliar_lp()

        # synthetic are in the last columns, except for the last one, which is b
        start_synthetic = self.tableau.shape[1] - self.slack_variables_added - 1
        self.tableau = np.delete(self.tableau, np.s_[start_synthetic: start_synthetic + self.slack_variables_added],
                                 axis=1)

        if simulate_auxiliar_operations_in_c:
            self.__restore_original_c()
        else:
            self.tableau[0] = self.old_c

        # after c is reinserted we need to redo the canonical form (make the basis columns canonical)
        if return_in_canonical:
            self.tableau = Simplex.putInCanonicalForm(self.tableau)

        return self.tableau

    def __change_objective_function(self):
        # tableau tem m (vero) + n (variaveis) + m (folgas, opcionais) + 1 de largura

        # here we have vero, variables, slack, new variables and b. They all should be zero
        new_c = np.zeros(self.tableau.shape[1] + self.slack_variables_added)

        # make the obj value for synthetic columns 1
        slack_start = self.tableau.shape[1] - 1
        columns = range(slack_start, slack_start + self.slack_variables_added)

        for c_i in columns:
            new_c[c_i] = 1

        return new_c

    def __change_restrictions_with_new_variables(self):

        # remove first row from tableau
        abMatrix = np.delete(self.tableau, 0, 0)

        # create n_restrictions * slack_variables_added zero, which will be square im my current implementation
        identityRestrictions = np.zeros((self.n_restrictions, self.slack_variables_added))

        # index of last column, before b
        offset = self.tableau.shape[1] - 1
        self.auxiliary_columns = list(range(offset, offset + self.slack_variables_added))

        abMatrix = np.insert(abMatrix, offset, identityRestrictions, axis=1)

        # insert in basis. I'm doing it this way as we may not need entire identity,
        # just the desired columns and indexes
        for row_index, column_index in enumerate(self.auxiliary_columns):
            if column_index != -1:
                abMatrix[row_index][column_index] = 1

        return abMatrix

    def pre_solve_auxiliar_problem(self):

        # TODO: Multiply by *1 here, not in matrix input
        self.tableau = self.__fix_negative_b_restrictions()

        # check to see if we need to insert synthetic variables, will almost never in Ax<=b problem
        new_tableau = self.__add_variables_to_auxiliary_problem().copy()

        # pivotear cada coluna da base auxiliar para colocar o c_i 0 zero
        for i, col in enumerate(self.auxiliary_columns):
            if col == -1:
                continue
            variable_index = i + 1
            new_tableau = Simplex.pivotTableau(new_tableau, col, variable_index)

        return new_tableau

    def __add_variables_to_auxiliary_problem(self):

        self.slack_variables_added = self.n_restrictions

        # I will not do this as i'm adding a full identity, but we could be more efficient
        # this behaviour can be achived by checking which variables do not have a
        # slack added (index of the basis is below the slack start
        # and add then, remebering to flag this addition in the object properties
        # trivial_basis = LinearAlgebra.findBasicColumns(self.tableau, drop_c=True, get_rightmost=True)
        # check missing identities and perform auxiliary on them...

        new_c = self.__change_objective_function()
        new_ab = self.__change_restrictions_with_new_variables()
        self.old_c = self.tableau[0]
        stacked_tableau = np.vstack((new_c, new_ab))
        return stacked_tableau

    def is_unfeasible(self):

        result = self.tableau[0][-1]
        # se o resultado for 0, é otimo.
        # TODO: Ver caso do livro do Thie, que o Scipy resolve
        if LinearAlgebra.equal_to_zero(result):
            basic_variables = LinearAlgebra.findBasicColumns(self.tableau, drop_c=True)
            logging.debug("basic_variables", basic_variables)

            for i, x_index in enumerate(basic_variables):
                # means that the variable in the basis is a synthetic variable
                if self.__is_synthetic_variable(x_index):
                    print(f"x index {x_index} is synthetic, ie, greater or equal than {self.m_variables}")
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

    def __is_synthetic_variable(self, x_index):
        return x_index in self.auxiliary_columns

    def __fix_negative_b_restrictions(self):
        """
        Fixes negative b rows, as we are not using dual simplex method
        :return: modified ab_matrix with only positive b values
        """
        ab_matrix = self.tableau.copy()

        for i in range(1, len(ab_matrix)):

            row = ab_matrix[i]
            b_i = row[-1]

            is_zero = np.isclose([b_i], [0])

            if not is_zero and b_i < 0:
                ab_matrix[i] = row * -1

        return ab_matrix
