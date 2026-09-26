import ast

def parse_csi(csi_string):
    """
    Extracts the CSI list from a CSI_Data string.

    Returns:
        list : Parsed CSI values
        None : If parsing fails
    """

    try:
        # Find the CSI array inside the string
        start = csi_string.rfind("[")
        end = csi_string.rfind("]") + 1

        # If brackets are missing
        if start == -1 or end == 0:
            print("\nParsing Error: '[' or ']' not found.")
            print("CSI String:")
            print(csi_string)
            return None

        # Extract only the list portion
        csi_list = csi_string[start:end]

        # Convert string to Python list
        values = ast.literal_eval(csi_list)

        # Ensure it is actually a list
        if not isinstance(values, list):
            print("\nParsing Error: Parsed data is not a list.")
            return None

        return values

    except Exception as e:
        print("\n========== PARSING ERROR ==========")
        print("Error:", e)
        print("CSI String:")
        print(csi_string)
        print("===================================")
        return None