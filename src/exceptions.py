class UnboundedError(Exception):
    def __init__(self, certificate=None):
        if certificate is not None:
            self.certificate = certificate
        else:
            raise TypeError("Exception must have certificate")

    def __str__(self):

        if self.certificate:
            return str(self.certificate)
        else:
            return 'Ilimitada'


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
