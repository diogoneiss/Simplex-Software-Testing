import logging
import numpy as np

from simplex import Simplex
from Utils.linear_algebra import LinearAlgebra
from exceptions import UnfeasibleError


class AuxiliarLP:
    def __init__(self, tableau, m_variables, n_restrictions):

        self.tableau = tableau
        self.m_variables = m_variables
        self.n_restrictions = n_restrictions

        # synthetic variables
        self.new_m_variables = m_variables + n_restrictions

        self.old_c = tableau[0]

    def __run_auxiliar_lp(self):
        canonical_tableau = self.setup_canonical_form()

        # create simplex object with the new tableau and variables
        simplexObj = Simplex(m=self.new_m_variables, n=self.n_restrictions, tableau=canonical_tableau)

        self.tableau = simplexObj.solve()

        # if 0 value for objective function not found it is unfeasible
        if self.is_unfeasible():
            raise UnfeasibleError

    def phase_1(self):
        """Checks for feasibility and returns a tableau with trivial basis

        Returns:
            _type_: _description_
        """

        # run simplex to try to get to zero value objective function
        # throws exception if unfeasible
        self.__run_auxiliar_lp()

        # remove synthetic columns (2*n+m to 3n+m) and restore c
        start_synthetic = self.m_variables + 2 * self.n_restrictions
        self.tableau = np.delete(self.tableau, np.s_[start_synthetic: start_synthetic + self.n_restrictions], axis=1)

        self.__restore_original_c()

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
        if result == 0:
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
