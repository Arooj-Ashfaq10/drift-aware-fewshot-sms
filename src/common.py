"""Shared constants: prompt template (follows SpaLLM-Guard, Salman et al. 2025, Listings 1-2)."""

ZS_SYSTEM = (
    "Act as an SMS spam classifier. Your task is to classify whether a given SMS is "
    "'Spam' or 'Ham' (benign). Do not reprint the SMS or provide any explanation; "
    "simply respond with 'Spam' for spam messages or 'Ham' for benign messages."
)


def build_system_prompt(examples):
    """examples: list of (text, label) with label 1 = spam, 0 = ham. Empty list -> zero-shot prompt."""
    if not examples:
        return ZS_SYSTEM
    lines = [ZS_SYSTEM, "Below are some examples of 'spam' and 'ham' Messages:"]
    for i, (t, y) in enumerate(examples):
        t = " ".join(str(t).split())
        lead = "For example: the output for" if i == 0 else "The output for"
        lines.append(f"{lead} '{t}' is {'Spam' if int(y) == 1 else 'Ham'}.")
    return "\n".join(lines)
