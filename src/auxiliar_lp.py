import logging
import math

import numpy as np

from simplex import Simplex
from Utils.linear_algebra import LinearAlgebra
from exceptions import UnfeasibleError


class AuxiliarLP:
    def __init__(self, tableau):

        self.tableau = tableau
        
        self.m_variables = LinearAlgebra.get_number_of_m_variables(tableau)
        self.n_restrictions = LinearAlgebra.get_number_of_n_restrictions(tableau)

        # synthetic variables
        self.new_m_variables =  self.m_variables + self.n_restrictions

        self.old_c = tableau[0]

    def __run_auxiliar_lp(self):

        # put sythetic columns in canonical form
        canonical_tableau = self.setup_canonical_form()

        # create simplex object with the new tableau and variables
        simplexObj = Simplex(m=self.new_m_variables, n=self.n_restrictions, tableau=canonical_tableau)

        # run simplex
        self.tableau = simplexObj.solve()

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

        # remove synthetic columns (2*n+m to 3n+m) and restore c
        start_synthetic = self.m_variables + 2 * self.n_restrictions
        self.tableau = np.delete(self.tableau, np.s_[start_synthetic: start_synthetic + self.n_restrictions], axis=1)

        if SIMULATE_AUXILIAR_OPERATIONS_IN_C:
            self.__restore_original_c()
        else:
            self.tableau[0] = self.old_c

        # after c is reinserted we need to redo the canonical form (make the basis columns canonical)
        self.tableau = Simplex.putInCanonicalForm(self.tableau)

        return self.tableau

    def __create_synthetic_c(self):
        # tableau tem m (vero) + n (variaveis) + m (folgas) + 1 de largura
        # vamos inserir uma coluna identidade e zerar o c

        zeroC = np.zeros(self.m_variables + 2 * self.n_restrictions)

        auxiliarC = np.ones(self.n_restrictions)

        # finalizar formato (0, 0, 0... 1, 1 ... 0)

        tmpC = np.hstack((zeroC, auxiliarC))

        fullC = np.hstack((tmpC, [0]))

        return fullC

    def __add_synthetic_restrictions(self, newC):

        # remove first row from tableau
        abMatrix = np.delete(self.tableau, 0, 0)

        # create n_restrictions * n_restrictions identity
        identityRestrictions = np.identity(self.n_restrictions)

        # insert just before b
        position = self.m_variables + 2 * self.n_restrictions

        tableauAb = np.insert(abMatrix, position, identityRestrictions, axis=1)

        fullTableau = np.vstack((newC, tableauAb))

        return fullTableau

    def setup_canonical_form(self):
        newC = self.__create_synthetic_c()
        newTableau = self.__add_synthetic_restrictions(newC)
        start = self.m_variables + 2 * self.n_restrictions

        # pivotear cada coluna da base trivial para colocar o 0 zero
        for i in range(0, self.n_restrictions):
            newTableau = Simplex.pivotTableau(newTableau, start + i, i + 1)

        return newTableau

    def __is_synthetic_variable(self, index):
        return index >= (self.m_variables + 2 * self.n_restrictions)

    def is_unfeasible(self):

        result = self.tableau[0][-1]

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
