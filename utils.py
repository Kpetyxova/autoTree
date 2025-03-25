import json
import re
import pandas as pd
from typing import Dict, Any


def extract_json_from_text(output: str) -> Dict[str, Any]:
    """
    Extract JSON content from text that may be wrapped in a code block.

    Args:
        output: String that may contain JSON content, possibly within a code block

    Returns:
        Parsed JSON object as a dictionary

    Example:
        >>> text = '''I propose the following feedback:
        ... ```json
        ... {"foo": 1}
        ... ```'''
        >>> extract_json_from_text(text)
        {'foo': 1}
    """
    match = re.search(r"```json*\n(.+?)```", output, re.MULTILINE | re.IGNORECASE | re.DOTALL)
    code_block_content = match.group(1) if match else output

    try:
        json_object = json.loads(code_block_content, strict=False)
        return json_object
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON content: {e}")


def check_if_label_in_data(split_group: Dict[str, Any], data: pd.DataFrame) -> bool:
    """
    Check if all single-element data labels in a split group exist in the DataFrame.

    Args:
        split_group: Dictionary containing group definitions with data labels
        data: DataFrame containing a 'Full labels' column to check against

    Returns:
        True if all single-element data labels exist in DataFrame, False otherwise
    """
    full_labels = set(data["Full labels"])

    for group in split_group.get("groups", []):
        group_data = group.get("data", [""])
        if len(group_data) == 1 and group_data[0] not in full_labels:
            return False

    return True
