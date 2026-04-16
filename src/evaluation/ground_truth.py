import json
from collections import Counter


def get_ground_truth_location():

    with open("data/processed/documents.json", "r") as f:
        docs = json.load(f)

    locations = []

    for text in docs:
        for line in text.split("\n"):
            if "LSOA name:" in line:
                loc = line.split("LSOA name:")[-1].strip()

                if loc and loc.lower() not in ["unknown", ""]:
                    locations.append(loc)

    if locations:
        return Counter(locations).most_common(1)[0][0]

    return None