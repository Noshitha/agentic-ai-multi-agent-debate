# grpo_tf/prompts.py
PROMPT_TEMPLATE = """{instruction}

Text:
{text}

You are an expert clinician. Follow this strict format:
Reasoning: <one or two sentences explaining your decision>
Label: <Present | Past | None>
"""

PROBLEM_WITH_EXPERIENCE = """Using prior experiences:
{experiences}

Now answer this task:
""" + PROMPT_TEMPLATE
