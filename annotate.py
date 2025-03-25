from prompts.prompt_annotate_dialogs import ANNOTATE_DIALOGS_SYSTEM, ANNOTATE_DIALOGS_USER
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import argparse
import os
from dotenv import load_dotenv
import pandas as pd
import json
from typing import List, Dict, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dialogs_path", type=str, help="Path to input dialogs file")
parser.add_argument("-t", "--tree_path", type=str, help="Path to decision tree file")
parser.add_argument("-m", "--model", type=str, default="gpt-4o", help="OpenAI model to use")
parser.add_argument("-b", "--binary", type=bool, default=False, help="Whether to use binary tree")
parser.add_argument("-o", "--output_dir", type=str, help="Output directory for results")
args = parser.parse_args()

dialogs_path = args.dialogs_path
tree_path = args.tree_path
model = args.model
binary = args.binary
output_dir = args.output_dir

# Set up LLM chain
prompt_annotator = ChatPromptTemplate.from_messages(
    [("system", ANNOTATE_DIALOGS_SYSTEM), ("human", ANNOTATE_DIALOGS_USER)]
)

llm_annot = ChatOpenAI(model=model, temperature=0.4)
chain_annotator = LLMChain(llm=llm_annot, prompt=prompt_annotator)


def get_question(current_node: Dict[str, Any], path: List[str]) -> Tuple[Dict[str, Any], str, List[str]]:
    """
    Get the current question and possible answers based on position in decision tree.

    Args:
        current_node: Current node in the decision tree
        path: List of keys defining path through tree to current position

    Returns:
        Tuple containing:
        - Current node dictionary
        - Question string to ask
        - List of possible answer strings
    """
    # Traverse to current node
    if path:
        for key in path:
            current_node = current_node[key]

    # Get possible answers for current node
    possible_answers = [
        f"Answer {i+1}: {current_node['groups'][i]['label']}" for i in range(len(current_node["groups"]))
    ]
    return current_node, current_node["question_to_define_groups"], possible_answers


def iterate_over_questions_binary(
    tree: Dict[str, Any], path: List[str], previous_context: str, current_utterance: str
) -> int:
    """
    Recursively traverse binary decision tree to classify utterance.

    Args:
        tree: Full decision tree dictionary
        path: Current path through tree
        previous_context: Previous dialog context
        current_utterance: Utterance to classify

    Returns:
        Integer label classification
    """
    current_node, question, possible_answers = get_question(tree, path)
    llm_inputs = {
        "previous_context": previous_context,
        "current_utterance": current_utterance,
        "question": question,
        "possible_answers": possible_answers,
    }

    # Try classification up to 2 times
    for _ in range(2):
        output = chain_annotator.invoke(llm_inputs)
        output = output.get("text")

        # Handle Answer 1
        if "answer 1" in output.lower():
            if current_node.get("next_split_group_1", {}):
                path.append("next_split_group_1")
                return iterate_over_questions_binary(tree, path, previous_context, current_utterance)
            logger.info(f"Group 1: {current_node['group_1_data']}")
            return current_node["group_1_data"][0]

        # Handle Answer 2
        elif "answer 2" in output.lower():
            if current_node.get("next_split_group_2", {}):
                path.append("next_split_group_2")
                return iterate_over_questions_binary(tree, path, previous_context, current_utterance)
            logger.info(f"Group 2: {current_node['group_2_data']}")
            return current_node["group_2_data"][0]

    # If no valid answer after retries, return default
    logger.info("Warning: No valid classification found")
    return current_node["group_2_data"][0]


def iterate_over_questions_non_binary(
    tree: Dict[str, Any], path: List[str], previous_context: str, current_utterance: str
) -> int:
    """
    Recursively traverse non-binary decision tree to classify utterance.

    Args:
        tree: Full decision tree dictionary
        path: Current path through tree
        previous_context: Previous dialog context
        current_utterance: Utterance to classify

    Returns:
        Integer label classification
    """
    current_node, question, possible_answers = get_question(tree, path)
    llm_inputs = {
        "previous_context": previous_context,
        "current_utterance": current_utterance,
        "question": question,
        "possible_answers": possible_answers,
    }

    output = chain_annotator.invoke(llm_inputs)
    output = output.get("text")

    # Check each possible answer
    for i in range(len(possible_answers)):
        if f"answer {i+1}" in output.lower():
            if current_node["groups"][i].get("next_split", {}):
                path.extend(["groups", i, "next_split"])
                return iterate_over_questions_non_binary(tree, path, previous_context, current_utterance)
            logger.info(f"Group {i+1}: {current_node['groups'][i]['data']}")
            return current_node["groups"][i]["data"][0]

    # If no valid answer found
    logger.info("Warning: No valid classification found")
    return current_node["groups"][0]["data"][0]


def main() -> None:
    """
    Main function to process dialog file and generate annotations.
    Reads input dialogs, processes each utterance through decision tree,
    and saves annotated results.
    """
    annotations_list = []

    # Load dialogs file
    if dialogs_path.endswith(".csv"):
        dialogs = pd.read_csv(dialogs_path)
    elif dialogs_path.endswith(".json"):
        dialogs = pd.read_json(dialogs_path)
    elif dialogs_path.endswith(".tsv"):
        dialogs = pd.read_csv(dialogs_path, sep="\t")
    else:
        raise ValueError(f"Invalid file type: {dialogs_path}")

    # Load decision tree
    with open(tree_path, "r") as f:
        questions_tree = json.load(f)

    # Process each dialog
    dialog_id_prev = None
    previous_speaker = None
    previous_text = None

    for _, utt in dialogs.iterrows():
        dialog_id = utt["dialog_id"]
        logger.info(f"Dialog ID: {dialog_id}")

        speaker = utt["speaker"]
        text = utt["text"]
        path = []
        logger.info(f"{speaker}: {text}")

        # Set previous context
        if dialog_id_prev != dialog_id:
            previous_context = "There is no previous context, as this is a beginning of the dialog."
            dialog_id_prev = dialog_id
        else:
            previous_context = f"Speaker {previous_speaker}: {previous_text}"
            

        # Get initial question
        _, question, possible_answers = get_question(questions_tree, [])
        llm_inputs = {
            "previous_context": previous_context,
            "current_utterance": f"Speaker {speaker}: {text}",
            "question": question,
            "possible_answers": possible_answers,
        }

        # Get initial classification
        output = chain_annotator.invoke(llm_inputs)
        output = output.get("text")

        # Set path based on classification
        for i in range(len(possible_answers)):
            if f"answer {i+1}" in output.lower():
                if not binary:
                    if questions_tree["groups"][i].get("next_split", {}):
                        path.extend(["groups", i, "next_split"])
                    break  # Add break to prevent further iterations
                else:
                    if i == 0:  # answer 1
                        if questions_tree.get("next_split_group_1", {}):
                            path.append("next_split_group_1")
                    elif i == 1:  # answer 2
                        if questions_tree.get("next_split_group_2", {}):
                            path.append("next_split_group_2")
                    break  # Add break to prevent further iterations

        current_utterance = f"Speaker {speaker}: {text}"
        if binary:
            label = iterate_over_questions_binary(questions_tree, path, previous_context, current_utterance)
        else:
            label = iterate_over_questions_non_binary(questions_tree, path, previous_context, current_utterance)

        # Update previous utterance info
        previous_speaker = speaker
        previous_text = text
        annotations_list.append(label)


    # Save results
    dialogs["Annotations"] = annotations_list
    os.makedirs(output_dir, exist_ok=True)
    dialogs.to_csv(f"{output_dir}/dialogs_annotated.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
