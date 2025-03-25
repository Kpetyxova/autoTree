import argparse
import pandas as pd
import os
from sklearn.metrics import classification_report

# Set up argument parser with required arguments and help text
parser = argparse.ArgumentParser(description="Evaluate annotation predictions against gold labels")
parser.add_argument("-d", "--dialogs_path", type=str, required=True, help="Path to dialogs file with gold annotations")
parser.add_argument("-a", "--annotations_path", type=str, required=True, help="Path to predicted annotations file")
parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save evaluation results")
args = parser.parse_args()

# Extract paths from arguments
dialogs_path = args.dialogs_path
annotations_path = args.annotations_path
output_dir = args.output_dir

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data files
try:
    dialogs = pd.read_csv(dialogs_path, sep="\t")
    annotations = pd.read_csv(annotations_path, sep="\t")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Could not find input file: {e.filename}")
except pd.errors.EmptyDataError:
    raise ValueError("One or both input files are empty")

# Extract prediction and ground truth columns
try:
    annotations_pred = annotations["Annotations"]
    annotations_true = dialogs["Annotations_gold"]
except KeyError as e:
    raise KeyError(f"Missing required column: {e}")

# Verify equal lengths
if len(annotations_pred) != len(annotations_true):
    raise ValueError("Prediction and ground truth annotations have different lengths")

# Generate and save evaluation report
eval_path = os.path.join(output_dir, "eval.txt")
with open(eval_path, "w") as f:
    f.write(classification_report(annotations_true, annotations_pred))
