from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import os
from dotenv import load_dotenv
import json
from graphviz import Digraph
from copy import deepcopy
import argparse
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from sentence_transformers import CrossEncoder
import logging
from treelib import Tree
from typing import Dict, Tuple, Optional, Any

from prompts.prompt_yes_no import SPLIT_INTO_GROUPS_HUMAN
from prompts.prompt_yes_no import SPLIT_INTO_GROUPS_SYSTEM as prompt_yes_no
from prompts.prompt_free_form_binary import SPLIT_INTO_GROUPS_SYSTEM as prompt_free_form_binary
from prompts.prompt_free_form_non_binary import SPLIT_INTO_GROUPS_SYSTEM as prompt_free_form_non_binary
from prompts.prompt_freq_guided_free_form_non_binary import (
    SPLIT_INTO_GROUPS_SYSTEM as prompt_freq_guided_free_form_non_binary,
)
from prompts.prompt_yes_no_mrbench import SPLIT_INTO_GROUPS_SYSTEM as prompt_yes_no_mrbench
from prompts.prompt_scorer import SCORE_SPLITS_HUMAN
from prompts.prompt_scorer import SCORE_SPLITS_SYSTEM as prompt_scorer
from prompts.prompt_scorer_freq import SCORE_SPLITS_SYSTEM as prompt_scorer_freq
from prompts.prompt_add_label import ADD_LABEL_HUMAN, ADD_LABEL_SYSTEM

from utils import extract_json_from_text, check_if_label_in_data


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
os.environ["OPENAI_API_KEY"] = api_key

# Initialize LLM
llm_tree = ChatOpenAI(model="gpt-4", temperature=0.4)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Create taxonomy tree from data")
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="yes_no",
    choices=[
        "yes_no",
        "free_form_binary",
        "free_form_non_binary",
        "split_selection",
        "freq_guided_split_selection",
        "freq_guided_free_form_non_binary",
        "yes_no_mrbench",
    ],
    help="Configuration type for tree creation",
)
parser.add_argument("-t", "--taxonomy_path", type=str, required=True, help="Path to taxonomy file (csv, json, or tsv)")
parser.add_argument("-d", "--description_path", type=str, required=True, help="Path to description file")
parser.add_argument("-o", "--output_path", type=str, required=True, help="Path for output files")

args = parser.parse_args()

# Initialize prompt templates
prompt_add_label = ChatPromptTemplate.from_messages([("system", ADD_LABEL_SYSTEM), ("human", ADD_LABEL_HUMAN)])
chain_add_label = LLMChain(llm=llm_tree, prompt=prompt_add_label)

# Load NLI model if needed
if args.config in ["split_selection", "freq_guided_split_selection"]:
    model_nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")

# Select appropriate system prompt
system_prompts = {
    "yes_no": prompt_yes_no,
    "free_form_binary": prompt_free_form_binary,
    "free_form_non_binary": prompt_free_form_non_binary,
    "split_selection": prompt_free_form_non_binary,
    "freq_guided_free_form_non_binary": prompt_freq_guided_free_form_non_binary,
    "freq_guided_split_selection": prompt_freq_guided_free_form_non_binary,
    "yes_no_mrbench": prompt_yes_no_mrbench,
}
split_into_groups_system = system_prompts[args.config]

# Load taxonomy data
file_extensions = {
    ".csv": lambda x: pd.read_csv(x),
    ".json": lambda x: pd.read_json(x),
    ".tsv": lambda x: pd.read_csv(x, sep="\t"),
}

file_ext = os.path.splitext(args.taxonomy_path)[1]
if file_ext not in file_extensions:
    raise ValueError(f"Unsupported taxonomy file format: {args.taxonomy_path}")

taxonomy = file_extensions[file_ext](args.taxonomy_path)

# Initialize group splitter prompt
prompt_grpoups_splitter = ChatPromptTemplate.from_messages(
    [("system", split_into_groups_system), ("human", SPLIT_INTO_GROUPS_HUMAN)]
)
chain_group_splits = LLMChain(llm=llm_tree, prompt=prompt_grpoups_splitter)

# Initialize scorer if needed
if args.config in ["split_selection", "freq_guided_split_selection"]:
    llm_scorer = ChatOpenAI(model="gpt-4", temperature=0.4)
    scorer_system = prompt_scorer_freq if args.config == "freq_guided_split_selection" else prompt_scorer
    prompt_scorer_splits = ChatPromptTemplate.from_messages([("system", scorer_system), ("human", SCORE_SPLITS_HUMAN)])
    chain_scorer_splits = LLMChain(llm=llm_scorer, prompt=prompt_scorer_splits)

# Load description
try:
    with open(args.description_path, "r") as f:
        description = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"Description file not found: {args.description_path}")


def split_into_groups(description: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Split data into groups based on description using LLM

    Args:
        description: Text description of how to split the data
        data: DataFrame containing taxonomy data

    Returns:
        Dictionary containing split groups and metadata
    """
    llm_inputs = {"description": description, "taxonomy": data.to_string(index=False)}
    output = chain_group_splits.invoke(llm_inputs)
    output = output.get("text")
    output = extract_json_from_text(output)
    return output


def eval_splits(data: pd.DataFrame, splits: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate quality of proposed splits using LLM scorer

    Args:
        data: DataFrame containing taxonomy data
        splits: Dictionary of proposed splits to evaluate

    Returns:
        Dictionary containing scores for each split
    """
    try:
        llm_inputs = {"taxonomy": data.to_string(index=False), "splits": splits}
        output = chain_scorer_splits.invoke(llm_inputs)
        output = output.get("text")
        output = extract_json_from_text(output)
    except Exception as e:
        logging.error(f"Error evaluating splits: {e}")
        logging.debug(f"LLM input: {llm_inputs}")
        # Retry once
        output = chain_scorer_splits.invoke(llm_inputs)
        output = output.get("text")
        output = extract_json_from_text(output)
    return output


def get_best_split(scores: Dict[str, Any], contr_checks: Dict[str, int]) -> Tuple[int, Optional[str]]:
    """
    Find best split based on scores and contradiction checks

    Args:
        scores: Dictionary of scores for each split
        contr_checks: Dictionary of contradiction counts for each split

    Returns:
        Tuple of (best score, best split name)
    """
    scale = {"bad": 0, "good": 1, "great": 2}
    best_split = None
    best_score = -1
    best_contr = float("inf")

    for split, score in scores.items():
        curr_score = scale[score["score"]]
        curr_contr = contr_checks[split]

        if (curr_score > best_score) or (curr_score == best_score and curr_contr < best_contr):
            best_score = curr_score
            best_split = split
            best_contr = curr_contr

    return best_score, best_split


def check_all_labels(data: pd.DataFrame, splits: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check that all labels are assigned to groups and add missing ones

    Args:
        data: DataFrame containing taxonomy data
        splits: Dictionary of splits to check

    Returns:
        Updated splits dictionary with all labels assigned
    """
    all_labels = set(data["Labels"])

    for split_name, split in splits.items():
        split_labels = set()
        for group in split.get("groups", []):
            split_labels.update(group["data"])

        missing_labels = all_labels - split_labels
        if missing_labels:
            logging.info(f"Missing labels in split {split_name}: {missing_labels}")

            for label in missing_labels:
                label_data = data[data["Labels"] == label]
                question = split["question_to_define_groups"]
                possible_answers = "\n".join(f"{group['label']}" for group in split["groups"])

                llm_inputs = {"label": label_data, "question": question, "possible_answers": possible_answers}

                output = chain_add_label.invoke(llm_inputs)
                assigned_group = output.get("text")

                for group in split["groups"]:
                    if group["label"] == assigned_group:
                        logging.info(f"Adding label {label} to group {group['label']}")
                        group["data"].append(label)
                        break

    return splits


def contradiction_check(parent_label: Optional[str], splits: Dict[str, Any]) -> Dict[str, int]:
    """
    Check for contradictions between parent and child labels

    Args:
        parent_label: Label of parent node
        splits: Dictionary of splits to check

    Returns:
        Dictionary counting contradictions for each split
    """
    checked_splits = {"split_1": 0, "split_2": 0, "split_3": 0}

    if parent_label:
        for split_name, split in splits.items():
            for group in split.get("groups", []):
                group_label = group.get("label", "")
                scores = model_nli.predict([(parent_label, group_label)])
                label_mapping = ["contradiction", "entailment", "neutral"]
                nli_class = label_mapping[scores.argmax()]

                if nli_class == "contradiction":
                    checked_splits[split_name] += 1
                    logging.info(f"Contradiction found in split {split_name}: {parent_label} and {group_label}")

    return checked_splits


def iterate_over_tree_binary(data: pd.DataFrame, current_split: Dict[str, Any], description: str) -> Dict[str, Any]:
    """
    Recursively build binary tree by splitting data into groups

    Args:
        data: DataFrame containing taxonomy data
        current_split: Current node's split information
        description: Text description for splitting

    Returns:
        Dictionary containing complete tree structure
    """
    data_group_1 = data[data["Labels"].isin(current_split["group_1_data"])]
    data_group_2 = data[data["Labels"].isin(current_split["group_2_data"])]

    if len(data_group_1) > 1:
        current_split["next_split_group_1"] = split_into_groups(description, data_group_1)
        iterate_over_tree_binary(data_group_1, current_split["next_split_group_1"], description)

    if len(data_group_2) > 1:
        current_split["next_split_group_2"] = split_into_groups(description, data_group_2)
        iterate_over_tree_binary(data_group_2, current_split["next_split_group_2"], description)

    return current_split


def iterate_over_tree_non_binary(data: pd.DataFrame, current_split: Dict[str, Any], description: str) -> Dict[str, Any]:
    """
    Recursively build non-binary tree by splitting data into groups

    Args:
        data: DataFrame containing taxonomy data
        current_split: Current node's split information
        description: Text description for splitting

    Returns:
        Dictionary containing complete tree structure
    """
    for group in current_split["groups"]:
        data_group = data[data["Labels"].isin(group["data"])]
        if len(data_group) > 1:
            group["next_split"] = split_into_groups(description, data_group)
            iterate_over_tree_non_binary(data_group, group["next_split"], description)

    return current_split


def iterate_over_tree_split_selection(
    data: pd.DataFrame, current_split: Dict[str, Any], description: str, paths: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively build tree by selecting best splits at each level

    Args:
        data: DataFrame containing taxonomy data
        current_split: Current node's split information
        description: Text description for splitting
        paths: Dictionary tracking selected paths through tree

    Returns:
        Dictionary containing complete tree structure
    """
    logging.info("Starting iterate_over_tree_split_selection function")

    current_scores = current_split["scores"]
    current_contr_checks = current_split["contr_checks"]
    best_score, best_split = get_best_split(current_scores, current_contr_checks)

    logging.info(f"Best score: {best_score}, Best split: {best_split}")
    paths["next_split"] = best_split
    paths["groups"] = []

    for i, group in enumerate(current_split[best_split].get("groups", [])):
        # Initialize group path if needed
        while len(paths["groups"]) <= i:
            paths["groups"].append({"next_split": None, "groups": []})

        data_group = data[data["Labels"].isin(group.get("data", []))]
        if len(data_group) <= 1:
            logging.info("Group size <= 1, skipping further splits")
            continue

        splits = {}
        for j in range(1, 4):
            split = split_into_groups(description, data_group)
            if not check_if_label_in_data(split, data_group):
                logging.warning(f"Split {j} data not in dataset, retrying...")
                split = split_into_groups(description, data_group)
                if not check_if_label_in_data(split, data_group):
                    logging.error(f"Split {j} failed twice, skipping")
                    split = {}
            splits[f"split_{j}"] = split

        if not any(splits.values()):
            logging.error("All splits failed, skipping group")
            continue

        # Clean up splits
        for split in splits.values():
            if split:
                for key in ["question_1", "question_2", "question_3", "answer_1", "answer_2", "answer_3"]:
                    split.pop(key, None)

        splits_checked = check_all_labels(data_group, splits)
        scores = eval_splits(data_group, splits_checked)
        contr_checks = contradiction_check(group["label"], splits_checked)

        best_score, best_split = get_best_split(scores, contr_checks)
        logging.info(f"Best score for group {i}: {best_score}, Best split: {best_split}")

        if best_score in [-1, 0]:
            logging.info("Best score is bad, trying next best split")
            current_scores_copy = deepcopy(current_scores)
            current_contr_checks_copy = deepcopy(current_contr_checks)
            current_scores_copy.pop(best_split)
            current_contr_checks_copy.pop(best_split)

            _, next_best_split = get_best_split(current_scores_copy, current_contr_checks_copy)
            new_current_split = {
                next_best_split: deepcopy(splits_checked[next_best_split]),
                "scores": current_scores_copy,
                "contr_checks": current_contr_checks_copy,
            }

            current_split["next_split"] = new_current_split
            paths["groups"][i]["next_split"] = new_current_split

            iterate_over_tree_split_selection(data_group, new_current_split, description, paths["groups"][i])
        else:
            paths["groups"][i]["next_split"] = best_split
            logging.info("Assigning splits and scores to group")

            if "next_split" not in group:
                group["next_split"] = {}

            group["next_split"].update(
                {
                    "split_1": splits_checked["split_1"],
                    "split_2": splits_checked["split_2"],
                    "split_3": splits_checked["split_3"],
                    "scores": scores,
                    "contr_checks": contr_checks,
                }
            )

            iterate_over_tree_split_selection(data_group, group["next_split"], description, paths["groups"][i])

    return current_split


def create_best_tree(data: Dict[str, Any], paths: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create best tree by following selected paths

    Args:
        data: Dictionary containing tree data
        paths: Dictionary tracking selected paths

    Returns:
        Dictionary containing best tree structure
    """
    for i, group in enumerate(data.get("groups", [])):
        if "next_split" in group:
            next_best_group = paths["groups"][i].get("next_split", "")
            group["next_split"] = data["groups"][i]["next_split"][next_best_group]
            create_best_tree(data["groups"][i]["next_split"], paths["groups"][i])

    return data


def build_tree_binary(tree: Digraph, annotated_data: Dict[str, Any], root: str) -> Digraph:
    """
    Build graphviz visualization of binary tree

    Args:
        tree: Graphviz Digraph object
        annotated_data: Dictionary containing tree data
        root: Name of root node

    Returns:
        Updated Graphviz Digraph
    """
    # Handle group 1
    if annotated_data.get("next_split_group_1"):
        node_1 = annotated_data["next_split_group_1"]["question_to_define_groups"]
        tree.node(node_1, node_1, shape="rectangle")
        tree.edge(root, node_1, label=annotated_data["group_1_label"])
        build_tree_binary(tree, annotated_data["next_split_group_1"], node_1)
    elif annotated_data["group_1_data"]:
        node_1 = annotated_data["group_1_data"][0]
        tree.node(node_1, node_1, color="purple")
        tree.edge(root, node_1, label=annotated_data["group_1_label"])

    # Handle group 2
    if annotated_data.get("next_split_group_2"):
        node_2 = annotated_data["next_split_group_2"]["question_to_define_groups"]
        tree.node(node_2, node_2, shape="rectangle")
        tree.edge(root, node_2, label=annotated_data["group_2_label"])
        build_tree_binary(tree, annotated_data["next_split_group_2"], node_2)
    elif annotated_data["group_2_data"]:
        node_2 = annotated_data["group_2_data"][0]
        tree.node(node_2, node_2, color="purple")
        tree.edge(root, node_2, label=annotated_data["group_2_label"])

    return tree


def build_tree_non_binary(tree: Digraph, annotated_data: Dict[str, Any], root: str) -> Digraph:
    """
    Build graphviz visualization of non-binary tree

    Args:
        tree: Graphviz Digraph object
        annotated_data: Dictionary containing tree data
        root: Name of root node

    Returns:
        Updated Graphviz Digraph
    """
    for group in annotated_data["groups"]:
        if group.get("next_split"):
            node = group["next_split"]["question_to_define_groups"]
            tree.node(node, node)
            tree.edge(root, node, label=group["label"])
            build_tree_non_binary(tree, group["next_split"], node)
        else:
            for data in group["data"]:
                tree.node(data, data, color="purple")
                tree.edge(root, data, label=group["label"])

    return tree


def build_tree_binary_txt(
    tree: Tree, filtered_tree: Dict[str, Any], parent: Optional[str] = None, node_count: int = 0
) -> None:
    """
    Build text representation of binary tree

    Args:
        tree: Tree object for text representation
        filtered_tree: Dictionary containing tree data
        parent: Name of parent node
        node_count: Counter for generating unique node IDs
    """
    question = filtered_tree["question_to_define_groups"]
    if parent is None:
        tree.create_node(question, question)
    else:
        tree.create_node(question, question, parent=parent)

    # Handle group 1
    if filtered_tree.get("group_1_label"):
        label_group_1 = filtered_tree["group_1_label"]
        id_label_group_1 = node_count
        tree.create_node(label_group_1, id_label_group_1, parent=question)
        node_count += 1

        if "next_split_group_1" in filtered_tree:
            build_tree_binary_txt(
                tree, filtered_tree["next_split_group_1"], parent=id_label_group_1, node_count=node_count
            )
        elif "group_1_data" in filtered_tree:
            for node in filtered_tree["group_1_data"]:
                tree.create_node(node, node_count, parent=id_label_group_1)
                node_count += 1

    # Handle group 2
    if filtered_tree.get("group_2_label"):
        label_group_2 = filtered_tree["group_2_label"]
        id_label_group_2 = node_count
        tree.create_node(label_group_2, id_label_group_2, parent=question)
        node_count += 1

        if "next_split_group_2" in filtered_tree:
            build_tree_binary_txt(
                tree, filtered_tree["next_split_group_2"], parent=id_label_group_2, node_count=node_count
            )
        elif "group_2_data" in filtered_tree:
            for node in filtered_tree["group_2_data"]:
                tree.create_node(node, node_count, parent=id_label_group_2)
                node_count += 1


def build_tree_non_binary_txt(tree: Tree, filtered_tree: Dict[str, Any], parent: Optional[str] = None) -> None:
    """
    Build text representation of non-binary tree

    Args:
        tree: Tree object for text representation
        filtered_tree: Dictionary containing tree data
        parent: Name of parent node
    """
    question = filtered_tree["question_to_define_groups"]
    if parent is None:
        tree.create_node(question, question)
    else:
        tree.create_node(question, question, parent=parent)

    for group in filtered_tree.get("groups", []):
        label = group["label"]
        tree.create_node(label, label, parent=question)

        if "next_split" in group:
            build_tree_non_binary_txt(tree, group["next_split"], parent=label)
        elif "data" in group:
            for node in group["data"]:
                tree.create_node(node, node, parent=label)


def create_tree_binary(data: pd.DataFrame, description: str) -> Dict[str, Any]:
    """
    Create complete binary tree structure

    Args:
        data: DataFrame containing taxonomy data
        description: Text description for splitting

    Returns:
        Dictionary containing complete tree structure
    """
    current_split = split_into_groups(description, data)
    annotated_data = iterate_over_tree_binary(data, current_split, description)

    with open(args.output_path, "w") as f:
        json.dump(annotated_data, f, indent=2)

    # Create visualization
    tree = Digraph()
    root = annotated_data["question_to_define_groups"]
    tree.node(root, root)
    build_tree_binary(tree, annotated_data, root)
    tree.render(args.output_path.replace(".json", ""), format="png", cleanup=True)

    # Create text representation
    tree_txt = Tree()
    build_tree_binary_txt(tree_txt, annotated_data)
    with open(args.output_path.replace(".json", ".txt"), "w") as f:
        f.write(tree_txt.show())

    return annotated_data


def create_tree_non_binary(data: pd.DataFrame, description: str) -> Dict[str, Any]:
    """
    Create complete non-binary tree structure

    Args:
        data: DataFrame containing taxonomy data
        description: Text description for splitting

    Returns:
        Dictionary containing complete tree structure
    """
    current_split = split_into_groups(description, data)
    annotated_data = iterate_over_tree_non_binary(data, current_split, description)

    # Save outputs
    with open(args.output_path, "w") as f:
        json.dump(annotated_data, f, indent=2)

    # Create visualization
    tree = Digraph()
    root = annotated_data["question_to_define_groups"]
    tree.node(root, root)
    build_tree_non_binary(tree, annotated_data, root)
    tree.render(args.output_path.replace(".json", ""), format="png", cleanup=True)

    # Create text representation
    tree_txt = Tree()
    build_tree_non_binary_txt(tree_txt, annotated_data)
    with open(args.output_path.replace(".json", ".txt"), "w") as f:
        f.write(tree_txt.show())

    return annotated_data


def create_tree_for_split_selection(data: pd.DataFrame, description: str) -> Dict[str, Any]:
    """
    Create tree by selecting best splits at each level

    Args:
        data: DataFrame containing taxonomy data
        description: Text description for splitting

    Returns:
        Dictionary containing best tree structure
    """
    paths = {}

    # Generate initial splits
    splits = {}
    for i in range(1, 4):
        split = split_into_groups(description, data)
        split = {
            k: v
            for k, v in split.items()
            if k not in ["question_1", "question_2", "question_3", "answer_1", "answer_2", "answer_3"]
        }
        splits[f"split_{i}"] = split

    # Process splits
    splits_checked = check_all_labels(data, splits)
    scores = eval_splits(data, splits_checked)
    contr_checks = contradiction_check(None, splits_checked)

    splits_checked.update({"scores": scores, "contr_checks": contr_checks})

    # Build tree
    annotated_data = iterate_over_tree_split_selection(data, splits_checked, description, paths)
    logging.info("Finished iterating over tree")

    # Create best tree
    next_split = paths["next_split"]
    best_tree = annotated_data[next_split]
    final_best_tree = create_best_tree(best_tree, paths)

    # Save outputs
    with open(args.output_path, "w") as f:
        json.dump(final_best_tree, f, indent=2)

    # Create visualization
    tree = Digraph()
    root = final_best_tree["question_to_define_groups"]
    tree.node(root, root)
    build_tree_non_binary(tree, final_best_tree, root)
    tree.render(args.output_path.replace(".json", ""), format="png", cleanup=True)

    # Create text representation
    tree_txt = Tree()
    build_tree_non_binary_txt(tree_txt, final_best_tree)
    with open(args.output_path.replace(".json", ".txt"), "w") as f:
        f.write(tree_txt.show())

    return final_best_tree


def main():
    """Main function to create taxonomy tree based on configuration"""
    if args.config in ["yes_no", "free_form_binary", "yes_no_mrbench"]:
        create_tree_binary(taxonomy, description)
    elif args.config == "free_form_non_binary":
        create_tree_non_binary(taxonomy, description)
    elif args.config in ["split_selection", "freq_guided_split_selection"]:
        create_tree_for_split_selection(taxonomy, description)


if __name__ == "__main__":
    main()
