import json
from collections import Counter


# 🔹 Load documents once (efficient)
def load_documents():
    with open("data/processed/documents.json", "r") as f:
        return json.load(f)


# 🔹 1. Most common LOCATION (LSOA-based)
def get_most_common_location():
    docs = load_documents()
    areas = []

    for text in docs:
        for line in text.split("\n"):
            if "LSOA name:" in line:
                loc = line.split("LSOA name:")[-1].strip()

                if loc and loc.lower() not in ["unknown", ""]:
                    areas.append(loc)

    if areas:
        return Counter(areas).most_common(1)[0][0]

    return "Not enough data"


# 🔹 2. Most common CRIME TYPE
def get_most_common_crime_type():
    docs = load_documents()
    crimes = []

    for text in docs:
        for line in text.split("\n"):
            if "Crime Type:" in line:
                crime = line.split("Crime Type:")[-1].strip()

                if crime and crime.lower() not in ["unknown", ""]:
                    crimes.append(crime)

    if crimes:
        return Counter(crimes).most_common(1)[0][0]

    return "Not enough data"


# 🔹 3. Most common OUTCOME
def get_most_common_outcome():
    docs = load_documents()
    outcomes = []

    for text in docs:
        for line in text.split("\n"):
            if "Outcome:" in line:
                outcome = line.split("Outcome:")[-1].strip()

                # Remove empty or unknown outcomes
                if outcome and outcome.lower() not in ["unknown", ""]:
                    outcomes.append(outcome)

    if outcomes:
        return Counter(outcomes).most_common(1)[0][0]

    return "Not enough data"