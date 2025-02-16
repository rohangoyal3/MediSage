import pandas as pd
import os

def search_names_in_xlsx(names_list):
    name_column = "Medicine Name"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Move one level up to reach the project root
    project_root = os.path.dirname(script_dir)

    # Construct the correct file path
    file_path = os.path.join(project_root, "data", "indian_medicine_data_cleaned.xlsx")

    data = pd.read_excel(file_path)
    # Ensure the name column exists
    if name_column not in data.columns:
        raise ValueError(f"Column '{name_column}' not found in the file.")

    # Filter rows where the name column matches any name in the list
    filtered_data = data[data[name_column].isin(names_list)]

    # Convert to dictionary format (using names as keys)
    result = {
        row[name_column]: row.to_dict() 
        for _, row in filtered_data.iterrows()
    }

    return result

