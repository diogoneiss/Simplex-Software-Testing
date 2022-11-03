import numpy as np
from Utils.linear_algebra import LinearAlgebra

DEBUG_TABLEAU = False


class TableauParsing:

    @staticmethod
    def read_n_m_dimensions():
        """ Le a primeira linha da entrada, que contém o número de restrições(M) e de variaveis (N)

        Returns:
            n_restricoes, m_variaveis: Tupla de inteiros com o número de restrições e o número de variáveis
        """

        # Read input from stdin
        dupla_n_m = tuple(map(int, input().split()))

        return dupla_n_m

    @staticmethod
    def __read_crude_c_ab_input(n_restrictions: int) -> tuple:
        """
        Reads c and ab, without numerical cleanup
        :param n_restrictions: number of restrictions
        :return: (c, ab)
        """

        # pegando o vetor c normal da linha
        array_c = np.array([input().strip().split()], float)
        # pegando a matriz AB normal, a cada n linha de restricao (restante)

        array_ab = np.array([input().strip().split()
                             for _ in range(n_restrictions)], int)

        return array_c, array_ab

    @staticmethod
    def __add_auxilliary_variables_to_a(ab: np.ndarray, n_restrictions: int, m_columns: int):
        """
        Adds slack variables to Ab matrix, such that it will become [A | I | b), with a n * n identity matrix
        :param ab:
        :param n_restrictions:
        :param m_columns:
        :return: tableau with [A | I | B] combined
        """

        if isinstance(ab, list):
            ab = np.array(ab, dtype=float)

        # insert at m column, last A column, an identity matrix
        position = m_columns
        tableau_base = np.insert(ab, position, np.identity(n_restrictions), axis=1)

        return tableau_base

    @staticmethod
    def __normalize_first_row(c: np.ndarray, n_restrictions: int):
        """
        Creates tableau first row from C.
        We need to make sure that the first row is [ -c | 0*n | 0)
            * C is negative because we are making a restriction C*x -w = 0, such that w is the value,
            which is equivalent to -C*x = w
            * We have n 0's, equivalent to every row, representing slack variables
            * Additional 0 as w restriction, which starts at 0, as simplex starts in trivial basis
        Exemple:
        [1, 2, 3], with n = 3 -> [-1, -2, -3 | 0, 0, 0 | 0]
        :param c: objective function, as 2d ndarray
        :param n_restrictions: number of restrictions
        :return: normalized c row with slack variables objective values
        """

        if type(c) == list:
            c = np.array(c)

        # sanity check as C cannot be 1-d in my login neither multirow
        # I prefer to fix it then to raise the exception
        if len(c.shape) != 2 or c.shape[0] != 1:
            # raise Exception(f"You messed up c dimensions, you must make it 2dimensional: 1 row with multiple columns. You have {c.shape}")
            c = c.reshape(1, -1)
        negated_c = np.negative(np.array(c))

        # I need to make this a 2d array, as C is 2d due to input parsing
        remaining_zeros = np.zeros(n_restrictions + 1).reshape(1, -1)

        first_row = np.hstack((negated_c, remaining_zeros))

        return first_row

    @staticmethod
    def __create_operations_register(n_restrictions: int):

        """ cria uma matriz com a primeira linha sendo de 0's e o restante sendo uma identidade
        para constituir o VERO. Formato:
        | 0 .... 0 0 0 |
        | 1 .... 0 0 0 |
        | 0 .... 1 0 0 |
        | 0 .... 0 1 0 |
        | 0 .... 0 0 1 |
        """

        # n width
        zeros = np.zeros(n_restrictions)
        # n * n
        identity = np.identity(n_restrictions)

        # colocar identidade depois da primeira linha
        # o vstack recebe uma TUPLA e não dois parametros, atenção nisso!
        operations_register = np.vstack((zeros, identity))

        return operations_register

    @staticmethod
    def __add_operations_register_tableau(tableau, n_restrictions: int):
        """
        Adds an operation register to the left side of tableau, see VERO in the documentation to understand its reason
        :param tableau: ab tableau
        :param n_restrictions:  number of restrictions
        :return: concatenated tableau with vero at beginning
        """

        operations_register = TableauParsing.__create_operations_register(n_restrictions)

        full_tableau = np.hstack((operations_register, tableau))

        return full_tableau

    @staticmethod
    def __fix_negative_b_restrictions(ab_matrix: np.ndarray):
        """
        Fixes negative b rows, as we are not using dual simplex method
        :param ab_matrix: matrix with just a and b
        :return: modified ab_matrix with only positive b values
        """

        for i, row in enumerate(ab_matrix):
            b_i = row[-1]

            is_zero = np.isclose([b_i], [0])

            if not is_zero and b_i < 0:
                ab_matrix[i] = row * -1

        return ab_matrix

    @staticmethod
    def read_everything_and_create_tableau():
        """
        Reads n, m, c and ab from input and create tableau
        :return:  n, m, full_tableau
        """
        n_restrictions, m_variables = TableauParsing.read_n_m_dimensions()
        c, ab = TableauParsing.__read_c_and_ab(n_restrictions)

        full_tableau = TableauParsing.create_full_tableau(c, ab, n_restrictions, m_variables)
        return n_restrictions, m_variables, full_tableau

    @staticmethod
    def read_ab_and_create_tableau(n_restrictions: int, m: int):
        """
        Reads c and ab  from input and create tableau
        :param n_restrictions:
        :param m: number of variables
        :return: full_tableau
        """

        # reads input and fixes negative b here
        c, ab = TableauParsing.__read_c_and_ab(n_restrictions)

        full_tableau = TableauParsing.create_full_tableau(c, ab, n_restrictions, m)

        return full_tableau

    @staticmethod
    def __read_c_and_ab(n_restrictions: int):
        """
        Reads c and ab from input
        :param n_restrictions: number of restrictions in tableau
        :return: c, ab_fixed
        """
        c, ab = TableauParsing.__read_crude_c_ab_input(n_restrictions)

        return c, ab

    @staticmethod
    def __create_c_ab_tableau(c: np.ndarray, a: np.ndarray, n_restrictions: int, m: int):
        tableau_base = TableauParsing.__add_auxilliary_variables_to_a(a, n_restrictions, m)

        first_line = TableauParsing.__normalize_first_row(c, n_restrictions)

        c_width = first_line.shape[1]
        ab_width = tableau_base.shape[1]

        if c_width != ab_width:
            raise Exception("You messed up AB and C shapes, they arent stackable")

        combined_c_ab = np.vstack((first_line, tableau_base))

        return combined_c_ab

    @staticmethod
    def create_full_tableau(c: np.ndarray, a: np.ndarray, n_restrictions: int, m: int) -> np.ndarray:
        """
        Creates tableau and adds operation register
        :param c: objective vector
        :param a: restriction matrix
        :param n_restrictions: number of restrictions
        :param m: number of vriables
        :return: generated tableau
        """

        combined_c_ab = TableauParsing.__create_c_ab_tableau(c, a, n_restrictions, m)

        full_tableau = TableauParsing.__add_operations_register_tableau(combined_c_ab, n_restrictions)

        # if b <0 we should multiply by -1
        full_tableau = TableauParsing.__fix_negative_b_restrictions(full_tableau)

        # sanity check, there should be no negative b values(last column)
        b_column = full_tableau.T[-1]

        if LinearAlgebra.any_below_zero(b_column):
            raise Exception(f"Negative b value inputed at b column {b_column}")
        return full_tableau
