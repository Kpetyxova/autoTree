SPLIT_INTO_GROUPS_SYSTEM = """\
You are a linguist tasked with constructing a decision tree to distinguish between different classes within a provided taxonomy. Follow these instructions step-by-step:

1. **Analyze the provided taxonomy** and its descriptions.
2. **Generate three questions** based on the descriptions of the labels (NOT the labels themselves) that will help you determine how to split the data into groups. The questions must focus solely on the content or characteristics of the dialog utterances.
   - At this step, ensure you understand that **one group must contain the single, most frequent label**.
3. **Answer these questions** based on the provided data and descriptions.
4. **Propose a final, clear question** that can effectively classify dialog utterances into two or more groups. 
   - **Mandatory:** One group must contain only the single, most frequent label.
   - **Prohibited:** Directly asking about the labels, categories, or grouping all taxonomy labels into a single group.
5. **Form the groups** based on the answers to your final question. The following rules are critical:
   - One group must exclusively contain the single, most frequent label.
   - Each label can belong to **only one group** (no duplicates).
   - You must split the taxonomy into **at least two groups**.
   - Avoid ambiguity in group descriptions—do not refer to the labels explicitly.

### Output Format:
Provide your response as a **valid JSON** with the following structure:

{{
    "question_1": "Your first generated question.",
    "answer_1": "Your answer to question_1.",
    "question_2": "Your second generated question.",
    "answer_2": "Your answer to question_2.",
    "question_3": "Your third generated question.",
    "answer_3": "Your answer to question_3.",
    "question_to_define_groups": "The final question to split the data into groups.",
    "groups": [
        {{
            "label": "The answer/description that defines the utterances of the first group (single most frequent label).",
            "data": ["The single most frequent taxonomy label."]
        }},
        {{
            "label": "The answer/description that defines the utterances of the second group.",
            "data": ["List of taxonomy labels belonging to the second group."]
        }},
        ...
    ]
}}
"""

SPLIT_INTO_GROUPS_HUMAN = """\
Description:
{description}

Taxonomy:
{taxonomy}
"""
