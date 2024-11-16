class RequiredAdditionalInfo(Exception):
    """
    Raised when table required additional info
    """
    def __init__(self):
        self.message = f"For table data need additional info (inn, kpp)"
        super().__init__(self.message)
