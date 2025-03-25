ANNOTATE_DIALOGS_SYSTEM = """\
Your task is to analyze a dialogue between two speakers and answer a specific question about the current utterance. \
Consider the relationship between the current utterance and the previous context when forming your answer."""


ANNOTATE_DIALOGS_USER = """\
Previous Context:
{previous_context}

Current Utterance:
{current_utterance}

Question: {question}
Possible Answers: {possible_answers}

Remember that the Question is about the Current Utterance.
You must select one answer from Possible Answers and reply only with "Answer 1", "Answer 2", etc., without any additional explanation."""
