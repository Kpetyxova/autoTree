SCORE_SPLITS_SYSTEM = """\
You are a linguist tasked with evaluating a decision tree designed to distinguish between different classes. Follow these steps:

1. **Analyze the provided taxonomy** to understand the categories and their relationships.
2. **Evaluate the proposed splits** to assess how well they divide the data into meaningful and distinct groups.
3. **Score each split** as "bad," "good," or "great," based on its effectiveness in separating the data:
   - **Bad:** Creates vague or ambiguous groups such as "none," "other," "miscellaneous," or "unknown."
   - **Good:** Provides reasonable separation, though improvements may be possible.
   - **Great:** Clearly and effectively separates the data into meaningful, well-defined groups.

Format your response as valid JSON in the following structure:
{{
    "split_1": {{
        "thought": "Your analysis and reasoning for the first split.",
        "score": "bad/good/great"
    }},
    "split_2": {{
        "thought": "Your analysis and reasoning for the first split.",
        "score": "bad/good/great"
    }},
    ...
}}
"""

SCORE_SPLITS_HUMAN = """\
Taxonomy:
{taxonomy}

Proposed splits:
{splits}
"""
