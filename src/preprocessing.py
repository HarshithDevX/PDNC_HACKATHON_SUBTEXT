import re

def clean_text(text: str) -> str:
    """Basic normalization for social and dialogue text."""
    if not isinstance(text, str):
        return ""
    # Normalize excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def truncate_dialogue_history(full_dialogue: str, max_turns: int = 5) -> str:
    """
    Truncation Strategy for long conversational context:
    Preserves the most recent turns leading up to the parent context and response.
    Earlier/older background turns are dropped first.
    """
    if not full_dialogue:
        return ""
    
    # Check if context contains turn delimiters
    if " [SEP] " in full_dialogue:
        turns = full_dialogue.split(" [SEP] ")
    elif "\n" in full_dialogue:
        turns = full_dialogue.split("\n")
    else:
        turns = [full_dialogue]
    
    # Retain the last `max_turns` turns
    if len(turns) > max_turns:
        turns = turns[-max_turns:]
    
    return " [SEP] ".join([t.strip() for t in turns if t.strip()])

def build_context_pair(row, strategy: str = "FULL_CONTEXT_RESPONSE"):
    """
    Constructs (context, response) pair according to the specified context strategy.
    
    Strategies:
    - RESPONSE_ONLY: Returns ("", response)
    - PARENT_RESPONSE: Returns (parent_context, response)
    - FULL_CONTEXT_RESPONSE: Returns (cleaned_full_dialogue_context, response)
    """
    response = clean_text(row.get("response", ""))
    parent = clean_text(row.get("parent_context", ""))
    full_ctx = clean_text(row.get("full_dialogue_context", ""))
    
    strategy = strategy.upper()
    
    if strategy == "RESPONSE_ONLY":
        context = ""
    elif strategy == "PARENT_RESPONSE":
        context = parent if parent else full_ctx
    elif strategy == "FULL_CONTEXT_RESPONSE":
        # Prioritize full dialogue; fall back to parent if empty
        if full_ctx:
            context = truncate_dialogue_history(full_ctx, max_turns=5)
        else:
            context = parent
    else:
        raise ValueError(f"Unknown context strategy: {strategy}")
    
    return context, response

def format_text_for_baseline(row, strategy: str = "FULL_CONTEXT_RESPONSE") -> str:
    """
    Formats the sample into a single string for linear/TF-IDF models.
    Concept: 'Context: <context> | Response: <response>'
    """
    context, response = build_context_pair(row, strategy)
    if not context:
        return response
    return f"Context: {context} | Response: {response}"
