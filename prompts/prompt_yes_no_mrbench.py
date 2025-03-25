SPLIT_INTO_GROUPS_SYSTEM = """\
You are a linguist building a decision tree to distinguish between different classes. Follow these steps:

1. Analyze the provided taxonomy.
2. Generate three questions that will help you understand how to split the data into two groups. These questions should focus on the descriptions of the labels, NOT on the labels themselves.
3. Answer these questions based on the data.
4. Propose a clear yes/no question to classify the dialog utterance into two groups: one group should contain a single label and the other group should contain all remaining labels.
5. **Do not** ask questions about labels or categories directly; focus solely on the content of the dialog utterances.
6. **Absolutely prohibited:** Grouping all taxonomy labels into one and naming the other group as "none," "other," "miscellaneous," "unknown," or similar.
7. The split must always involve two meaningful groups based on content.
8. Labels are always integers!
9. Feel free to use examples in question_to_define_groups, if they are helpful.

Format your response as a valid JSON:
{{
    "question_1": "Your first generated question.",
    "answer_1": "Your answer to question_1.",
    "question_2": "Your second generated question.",
    "answer_2": "Your answer to question_2.",
    "question_3": "Your third generated question.",
    "answer_3": "Your answer to question_3.",
    "question_to_define_groups": "The proposed Yes/No question to split the data into groups.",
    "group_1_label": "Yes/No",
    "group_2_label": "Yes/No",
    "group_1_data": ["Single label"] ### It must be of length 1!!!, labels are integers!
    "group_2_data": ["All other labels"] ### labels are integers!
}}

You must always split the taxonomy into two **meaningful** groups. Creating vague or catch-all groups, such as "none," "other," "miscellaneous," or "unknown," is **strictly prohibited** and will result in an incorrect response. The split must be content-based and meaningful."""

SPLIT_INTO_GROUPS_HUMAN = """\
Description:
{description}

Taxonomy:
{taxonomy}
"""
