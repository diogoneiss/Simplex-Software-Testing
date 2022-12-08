class UnboundedError(Exception):
    def __init__(self, certificate=None, x_solution=None):
        if certificate is not None:
            self.certificate = certificate
        else:
            raise TypeError("Exception must have certificate")
        if x_solution is not None:
            self.x_solution = x_solution
        else:
            raise TypeError("Exception must have x_solution")

    def __str__(self):
        return f"{str(self.x_solution)}\n{str(self.certificate)}"


class UnfeasibleError(Exception):
    def __init__(self, certificate=None):
        if certificate is not None:
            self.certificate = certificate
        else:
            raise TypeError("Exception must have certificate")

    def __str__(self):

        if self.certificate:
            return str(self.certificate)
        else:
            return 'Inviavel'
