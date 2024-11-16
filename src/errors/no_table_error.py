class NoTableError(Exception):
    """
    Raised when img2table doesn't detect table
    """
    def __init__(self, table_path):
        self.message = f"File: {table_path} doesn't contains table"
        super().__init__(self.message)
