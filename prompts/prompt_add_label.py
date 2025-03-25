ADD_LABEL_SYSTEM = """\
You are a linguist assisting in categorizing labels within a taxonomy.

Your task is to decide the appropriate group for a given label based on its description and example.
Analyze the label carefully and assign it to the most relevant group in the taxonomy.

"""

ADD_LABEL_HUMAN = """\
New Label:
{label}

Question to guide group assignment:
{question}

Possible Groups:
{possible_answers}

Respond with the name of the most appropriate group only, without providing any explanations or additional comments.
"""
