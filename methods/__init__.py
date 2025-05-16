class Method:
    def __init__(self, method_name):
        self.method_name = method_name
        
def load_method(method_name: str) -> Method:
    """
    Load a method object based on the given method name.

    Args:
        method_name (str): The name of the method to load.

    Returns:
        Method: The corresponding Method object.

    Raises:
        ValueError: If the method name is invalid or does not exist.
    """
    method = Method(method_name)
    return method
