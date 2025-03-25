import argparse
import pandas as pd
import json
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gold_path", type=str)
parser.add_argument("-a", "--annotations_path", type=str)
parser.add_argument("-o", "--output_dir", type=str)
parser.add_argument("-d", "--dialogs_path", type=str)
parser.add_argument("-l", "--level", type=str)
args = parser.parse_args()

gold_path = args.gold_path
annotations_path = args.annotations_path
output_dir = args.output_dir
dialogs_path = args.dialogs_path
level = args.level

annotations = pd.read_csv(annotations_path, sep="\t")
annotations_list = annotations["Annotations"].tolist()

with open(gold_path, "r") as f:
    all_gold_train = json.load(f)

dialogs = pd.read_csv(dialogs_path)
dialogs_ids = set(dialogs["dialog_id"].tolist())


def main():
    gold_annotaions_new = []
    pred_annotations_new = []
    for i, dialog_id in enumerate(dialogs_ids):
        gold_annotations = all_gold_train[str(dialog_id)]["good"]
        pred_annotation = annotations_list[i]
        for j, options in enumerate(gold_annotations):
            if pred_annotation is not None:
                if "Command" in pred_annotation:
                    pred_annotation = "Open.Command"
            if level == "first":
                options = [label.split(".")[0] for label in options]
            elif level == "second":
                options = [".".join(label.split(".")[:2]) for label in options]
            if pred_annotation is not None:
                pred_annotations_new.append(pred_annotation)
                if pred_annotation in options:
                    gold_annotaions_new.append(pred_annotation)
                else:
                    gold_annotaions_new.append(options[0])

    with open(f"{output_dir}/eval.txt", "w") as f:
        f.write(classification_report(gold_annotaions_new, pred_annotations_new))


if __name__ == "__main__":
    main()
