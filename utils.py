def discourse_effectiveness_to_int(discourse_effectiveness):
    l = discourse_effectiveness.lower()
    if l == "ineffective":
        return 0
    if l == "adequate":
        return 1
    if l == "effective":
        return 2
    raise ValueError(f"Unrecognized discourse effectiveness: {discourse_effectiveness}")


# Not used, but still useful to know
def label_number_to_label_name(x):
    if x == 1:
        return "Ineffective"
    if x == 0:
        return "Adequate"
    if x == 2:
        return "Effective"
    raise ValueError("Unknown label")
