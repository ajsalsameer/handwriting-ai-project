# backend/utils/suggester.py  ← OFFICIAL FINAL VERSION (Yours!)
def create_smart_instruction(missing_letters, missing_symbols):
    """
    Generates a 'Golden Sample' request that covers all key handwriting features.
    """
    priority_letters = ['a', 'b', 'e', 'f', 'g', 'h', 'j', 'm', 'q', 'r', 'y', 'z']
    
    combined = set(priority_letters + missing_letters)
    letters_to_ask = sorted(list(combined))
    
    word = "the quick brown fox" 

    instruction = f"""
**Perfect! To learn your handwriting style 100% accurately, please do this:**

1. Grab a blank piece of paper.
2. Write these **letters** clearly:
   **{'  '.join([l.upper() for l in letters_to_ask])}**

3. Write this **sentence** below them:
   **{word_to_ask}**

4. Take a **clear photo** and upload it below.
"""
    return instruction