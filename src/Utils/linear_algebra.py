import sys

import numpy as np
import logging


class LinearAlgebra:

    @staticmethod
    def retrive_certificate(tableau, n_restrictions):
        firstRow = tableau[0]
        vero_row = firstRow[:n_restrictions]

        return vero_row

    @staticmethod
    def get_x_solution(tableau):
        x_solution = LinearAlgebra.get_solution(tableau)
        m_variables = LinearAlgebra.get_number_of_m_variables(tableau)
        x_solutions_without_aux_variables = x_solution[:m_variables]

        return x_solutions_without_aux_variables

    @staticmethod
    def replace_values_smaller_then_tol(array):

        new_array = np.copy(array)
        should_be_zero = np.isclose(array, np.zeros(array.shape))
        new_array[should_be_zero] = 0

        return new_array

    @staticmethod
    def any_below_zero(array):
        replaced = LinearAlgebra.replace_values_smaller_then_tol(array)
        return np.any(replaced < 0)

    @staticmethod
    def all_below_zero(array):
        replaced = LinearAlgebra.replace_values_smaller_then_tol(array)
        return np.all(replaced < 0)

    @staticmethod
    def remove_equal_rows(ab: np.ndarray):
        """
        Remove equal rows from the tableau
        :param ab:
        :return:
        """

        tableau = np.copy(ab)
        removed_rows = []
        for i in range(0, tableau.shape[0]):
            for j in range(i + 1, tableau.shape[0]):
                if j in removed_rows or i == j:
                    continue
                # silence warnings of division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    quotient = np.divide(tableau[i], tableau[j])

                # check if all elements are equal
                if np.allclose(quotient, quotient[0]):
                    logging.debug(f"Removing row {j} from tableau")
                    logging.debug(f"Row {tableau[j]} is equal to row {tableau[i]}")
                    removed_rows.append(j)

        tableau = np.delete(tableau, removed_rows, axis=0)
        return tableau, removed_rows

    @staticmethod
    def smaller_than_zero(number: float):

        array = np.array([number])
        # fixes numeric precision
        return LinearAlgebra.any_below_zero(array)

    @staticmethod
    def equal_to_zero(number: float):
        array = np.array([number])
        # fixes numeric precision
        replaced = LinearAlgebra.replace_values_smaller_then_tol(array)
        return np.any(replaced == 0)

    @staticmethod
    def arrayPrint(array):
        """
        Desestrutura e printa o array bonitinho, separando em espaços e com arredondamento de 7 casas decimais
        """
        print(*array.round(7), sep=' ')

    @staticmethod
    def matprint(mat: np.ndarray, fmt="g"):
        """
        Pretty print a numpy matrix, ie, two dimensional array
        """
        if isinstance(mat, list):
            mat = np.array(mat)

        # if its one dimensional, print it as a vector using the arrayPrint function
        if len(mat.shape) == 1:
            LinearAlgebra.arrayPrint(mat)
            return

        col_maxes = [max([len(("{:" + fmt + "}").format(x))
                          for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end=", ")
            print("")

    @staticmethod
    def get_number_of_m_variables(tableau, has_vero=True):
        n_restrictions = LinearAlgebra.get_number_of_n_restrictions(tableau)

        # removing the b  column
        width = len(tableau[0]) - 1

        # if vero is present we need to remove extra n columns
        if has_vero:
            width -= (n_restrictions * 2)
        else:
            width -= n_restrictions

        return width

    @staticmethod
    def get_number_of_n_restrictions(tableau: np.ndarray):
        """
        Returns:
            int: The number of restrictions.
        """

        if isinstance(tableau, list):
            tableau = np.array(tableau)

        return tableau.shape[0] - 1

    @staticmethod
    def drop_vero(tableau: np.ndarray, n_restrictions=0):
        """
        Returns:
            np.ndarray: The tableau without the vero, which is the first n_restrictions columns.
        """

        if isinstance(tableau, list):
            tableau = np.array(tableau)

        if n_restrictions == 0:
            n_restrictions = LinearAlgebra.get_number_of_n_restrictions(tableau)

        return tableau[:, n_restrictions:]

    @staticmethod
    def get_solution(tableau: np.ndarray):
        """Returns the solution vector x for the given tableau
        """

        if isinstance(tableau, list):
            tableau = np.array(tableau)

        n_restrictions = LinearAlgebra.get_number_of_n_restrictions(tableau)
        x_width = LinearAlgebra.get_number_of_m_variables(tableau) + n_restrictions

        cleaned_tableau = LinearAlgebra.drop_vero(tableau, n_restrictions)

        basic_columns = LinearAlgebra.findBasicColumns(cleaned_tableau, drop_vero=False, drop_b=True)

        """
        If the basic column is [0 3 1], that means that x0 = b_0, x3 = b_1 and x1 = b_2
        The b index is basic_column index 
        """

        # every x out of basis is zero
        x = np.zeros(x_width)

        for i, column in enumerate(basic_columns):
            correct_row = i + 1
            b_value = tableau[correct_row, -1]
            x[column] = b_value
        return x

    @staticmethod
    def extract_feasible_columns(tableau: np.ndarray, remove_b=True) -> np.ndarray:
        """_summary_: Extracts the feasible columns from the tableau.

        Args:
            tableau (np.ndarray): tableau with vero, a, aditional variables and b
            remove_b (bool, optional): whether to remove the b column. Defaults to True.

        Returns:
            np.ndarray: sliced tableau with only the feasible columns
        """

        n_restrictions = LinearAlgebra.get_number_of_n_restrictions(tableau)

        if remove_b:
            return tableau[:, n_restrictions: -1]
        else:
            return tableau[:, n_restrictions:]

    @staticmethod
    def findBasicColumns(tableau, drop_c=False, drop_vero=True, drop_b=True, get_rightmost=False):
        """ Gets the column indexes of the basic columns in the tableau, that is, the columns with one 1 and all zeros
        Each index represents a restriction and the value represents the variable index that is in the basis

        """

        n_restrictions = LinearAlgebra.get_number_of_n_restrictions(tableau)

        basicIndexes = np.full(n_restrictions, -1)
        row_offset = 0
        column_offset = 0
        if drop_c:
            row_offset += 1
            tableau = np.delete(tableau, 0, axis=0)
        if drop_vero:
            column_offset += n_restrictions
            tableau = np.delete(tableau, np.s_[0: n_restrictions], axis=1)
        if drop_b:
            tableau = np.delete(tableau, -1, axis=1)

        for idx, column in enumerate(tableau.T):
            # skip operations register and b column if desired

            is_basis = np.count_nonzero(column) == 1 and np.sum(column) == 1

            # if c was not dropped, we need to check the canonical form ( c_i == 0)
            if not drop_c:
                is_basis = is_basis and column[0] == 0

            if is_basis:

                # finds the index of the 1 in the column
                # this is the b value and idx is the active X_idx
                position = column.tolist().index(1)

                # if c row is still present we need to decrease the position value
                if not drop_c:
                    position -= 1

                if basicIndexes[position] != -1:
                    # case where we find another valid basis in the end, most likely a slack column
                    if get_rightmost:
                        basicIndexes[position] = idx + column_offset
                    else:
                        # matprint(tableau)
                        logging.debug(
                            f"{idx}th column will not enter, already found basic row for {position}th restriction with column  X_{basicIndexes[position]}")
                else:
                    basicIndexes[position] = idx + column_offset

        return basicIndexes
