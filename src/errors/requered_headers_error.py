class RequiredHeadersError(Exception):
    """
    Raised when table required additional info
    """
    def __init__(self):
        self.message = f"For table data headers from previous table"
        super().__init__(self.message)
