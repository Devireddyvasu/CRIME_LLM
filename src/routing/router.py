# 🔹 ROUTE QUERY (decides pipeline)
def route_query(query):
    q = query.lower()

    if any(word in q for word in ["most", "highest", "common", "frequent", "number of", "how many", "rank", "top", "identify the"]):
        return "analytical"

    elif any(word in q for word in ["where", "location", "area"]):
        return "location based"

    else:
        return "general"

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

    # ✅ Aggregations / ODIN
    if any(word in query for word in ["number of", "how many"]):
        intents.append("count")
    if any(word in query for word in ["rank", "top 10", "top"]):
        intents.append("rank")

    # ✅ Geographies (ODIN)
    if any(word in query for word in ["lsoa"]):
        intents.append("lsoa")
    if any(word in query for word in ["ward"]):
        intents.append("ward")
    if any(word in query for word in ["output area"]):
        intents.append("output_area")
    if any(word in query for word in ["local authority"]):
        intents.append("local_authority")

    # 🔹 Fallback → if generic "most" query and no geography yet
    if not any(x in intents for x in ["location", "lsoa", "ward", "output_area", "local_authority"]) and any(word in query for word in ["most", "highest", "top"]):
        intents.append("location")

    return intents