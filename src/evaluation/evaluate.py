def normalize_answer(text):
    if text is None:
        return ""

    text = text.lower().strip()

    # Remove prefixes
    text = text.replace("the area with the most crimes is", "")
    text = text.replace("the most common crime type is", "")
    text = text.replace("the most common outcome is", "")

    # Remove unwanted text
    text = text.replace("# 🔥 add this", "")

    # Clean punctuation
    text = text.replace(".", "").strip()

    return text


def evaluate_answer(generated, expected):
    """
    Simple similarity score using exact match logic
    """

    gen_clean = normalize_answer(generated)
    exp_clean = normalize_answer(expected)

    # Exact match → score = 1
    if gen_clean == exp_clean:
        return 1.0

    # Partial match → simple overlap score
    gen_words = set(gen_clean.split())
    exp_words = set(exp_clean.split())

    if not exp_words:
        return 0.0

    overlap = len(gen_words & exp_words)
    score = overlap / len(exp_words)

    return score
