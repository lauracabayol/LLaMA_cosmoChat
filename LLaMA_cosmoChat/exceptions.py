"""Custom exceptions for the LLaMA_cosmoChat package."""

class QueryValidationError(Exception):
    """Exception raised when query validation fails.
    
    This exception is raised when a query does not meet the validation requirements,
    such as not referencing any available tables in the database.
    """
    pass

class SQLExtractionError(Exception):
    """Raised when SQL query cannot be extracted from the model's response."""
    pass

# Add other custom exceptions here as needed
class ModelInitializationError(Exception):
    """Exception raised when model initialization fails."""
    pass

class SchemaError(Exception):
    """Exception raised when there are issues with the database schema."""
    pass 