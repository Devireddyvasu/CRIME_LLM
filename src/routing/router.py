# 🔹 ROUTE QUERY (decides pipeline)
def route_query(query):
    query = query.lower()

    if any(word in query for word in ["most", "highest", "top"]):
        return "analytical_query"

    return "general_query"


# 🔹 DETECT MULTIPLE INTENTS
def detect_intents(query):
    query = query.lower()
    intents = []

    # ✅ Outcome (highest priority)
    if any(word in query for word in ["outcome", "result", "status"]):
        intents.append("outcome")

    # ✅ Crime type (specific phrases only)
    if any(word in query for word in ["crime type", "type of crime", "which crime"]):
        intents.append("crime_type")

    # ✅ Location
    if any(word in query for word in ["where", "location", "area"]):
        intents.append("location")

    # 🔹 Fallback → if generic "most" query
    if not intents and any(word in query for word in ["most", "highest", "top"]):
        intents.append("location")

    return intents