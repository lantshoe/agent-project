from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
       Evaluates a mathematical expression and returns the result.
       Use this whenever you need to perform calculations.
       Example input: "2 + 2", "100 * 3.14", "(5 + 3) * 2"
   """
    try:
        result = eval(expression, {"__builtins__":{}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"
