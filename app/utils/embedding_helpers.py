def format_qwen3_embedding_instruction(text, instruction=None):
    """
    Format the input for Qwen3 embedding.
    Args:
        text: The text to embed.
        instruction: The instruction to use for the embedding. If None, the text is returned as is.
    Returns:
        The formatted input for Qwen3 embedding.
    """
    if instruction is None:
        return text
    else:
        return f"Instruct: {instruction}\nText: {text}"