import numpy as np
from Utils.linear_algebra import LinearAlgebra

DEBUG_TABLEAU = False


class TableauParsing:

    @staticmethod
    def readDimensions():
        """ Le a primeira linha da entrada, que contém o número de restrições(M) e de variaveis (N)

        Returns:
            n_restricoes, m_variaveis: Tupla de inteiros com o número de restrições e o número de variáveis
        """

        # Read input from stdin
        dupla_n_m = tuple(map(int, input().split()))

        return dupla_n_m

    @staticmethod
    def readInput(n_restrictions: int) -> tuple:
        """Le o vetor C e a matriz AB

        Args:
            n (int): numero de restricoes

        Returns:
            arrayC, arrayAB: Tupla de arrays numpy com o vetor C e a matriz AB, ambos com duas dimenções
        """

        # pegando o vetor c normal da linha
        arrayC = np.array([input().strip().split()], float)
        # pegando a matriz AB normal, a cada n linha de restricao (restante)

        arrayAB = np.array([input().strip().split()
                            for _ in range(n_restrictions)], int)

        return arrayC, arrayAB

    @staticmethod
    def add_auxilliary_variables_to_a(a, n_restrictions: int, m: int):

        if isinstance(a, list):
            a = np.array(a)

        # inserir na posicao m (ultima variavel) uma matriz identidade n*n
        tableauBase = np.insert(a, m, np.identity(n_restrictions), axis=1)

        return tableauBase

    @staticmethod
    def mount_first_line(c, n_restrictions: int):
        """
        cria uma linha com -c e os 0's das variáveis de folga e do valor objetivo
        Exemplo:
        [1, 2, 3], com n = 3 -> [1, 2, 3, 0, 0, 0, 0]
        3 deles sao variaveis e 1 do v.o
        """
        opositeC = np.array(c) * -1

        # need n 0's and one 0 for objective function value
        firstLine = np.append(opositeC, np.zeros(n_restrictions + 1))
        return firstLine

    @staticmethod
    def create_operations_register(n_restrictions: int):

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
        operationsRegister = np.vstack((zeros, identity))

        return operationsRegister

    @staticmethod
    def create_regular_tableau(c, a, n_restrictions: int, m: int):
        tableauBase = TableauParsing.add_auxilliary_variables_to_a(a, n_restrictions, m)
        firstLine = TableauParsing.mount_first_line(c, n_restrictions)

        combinedC_AB = np.vstack((firstLine, tableauBase))

        return combinedC_AB

    @staticmethod
    def add_operations_register_tableau(regularTableau, n_restrictions: int):

        operationsRegister = TableauParsing.create_operations_register(n_restrictions)

        if DEBUG_TABLEAU:
            print("operationsRegister: ", operationsRegister.shape)
            LinearAlgebra.matprint(operationsRegister)
            print("Regular Tableau: ", regularTableau.shape)
            LinearAlgebra.matprint(regularTableau)

        fullTableau = np.hstack((operationsRegister, regularTableau))

        if DEBUG_TABLEAU:
            print("Full Tableau: ", fullTableau.shape)
            LinearAlgebra.matprint(fullTableau)

        return fullTableau

    @staticmethod
    def fix_negative_b_restrictions(tableau: np.ndarray):
        for i, row in enumerate(tableau):
            b_i = row[-1]
            if b_i < 0:
                tableau[i] = row * -1
        return tableau

    @staticmethod
    def readAndCreateTableau(n_restrictions: int, m: int):

        c, a = TableauParsing.readInput(n_restrictions)

        full_tableau = TableauParsing.createTableau(c, a, n_restrictions, m)
        full_tableau = TableauParsing.fix_negative_b_restrictions(full_tableau)
        return full_tableau

    @staticmethod
    def createTableau(c, a, n_restrictions: int, m: int):
        combinedC_AB = TableauParsing.create_regular_tableau(c, a, n_restrictions, m)
        fullTableau = TableauParsing.add_operations_register_tableau(combinedC_AB, n_restrictions)

        return fullTableau
