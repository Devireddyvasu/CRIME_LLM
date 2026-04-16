import pandas as pd
import json
import os

def preprocess_data(input_path, output_path):

    # Detect file type automatically (professional approach)
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".xlsx"):
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Unsupported file format")

    # Clean column names (VERY IMPORTANT)
    df.columns = df.columns.str.strip()

    print("Columns detected:", df.columns.tolist())

    # Handle missing values
    df = df.fillna("Unknown")

    # Convert rows to structured text
  
    def row_to_text(row):
        return f"""
        Crime ID: {row.get('Crime ID', '')}
        Month: {row.get('Month', '')}
        Reported by: {row.get('Reported by', '')}
        Falls within: {row.get('Falls within', '')}
        Longitude: {row.get('Longitude', '')}
        Latitude: {row.get('Latitude', '')}
        Location: {row.get('Location', '')}
        LSOA code: {row.get('LSOA code', '')}
        LSOA name: {row.get('LSOA name', '')}   
        Crime Type: {row.get('Crime type', '')}
        Outcome: {row.get('Last outcome category', '')}
        Context: {row.get('Context', '')}
    """

    documents = df.apply(row_to_text, axis=1).tolist()

    # Ensure folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(documents, f, indent=4)

    print(f"✅ Preprocessing complete! Saved to {output_path}")
    print(f"Total documents: {len(documents)}")


if __name__ == "__main__":
    preprocess_data(
        "data/raw/2025-12-cambridgeshire-street.csv",
        "data/processed/documents.json"
    )