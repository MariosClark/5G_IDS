import csv
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

# Base directory - adjust as needed or keep current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Task 1: Splitting Combined.csv into Benign and Attacks ===
print("Task 1: Splitting dataset into Benign and Attacks...")
input_file = os.path.join(BASE_DIR, "Combined.csv")
output_dir = os.path.join(BASE_DIR, "Datasets")
os.makedirs(output_dir, exist_ok=True)

benign_file = os.path.join(output_dir, "Benign.txt")
attacks_file = os.path.join(output_dir, "Attacks.txt")

df_full = pd.read_csv(input_file, low_memory=False)
print(f"Original rows: {len(df_full)}")

# Remove duplicates
df_full = df_full.drop_duplicates()
print(f"Rows after removing duplicates: {len(df_full)}")

with open(input_file, 'r') as csv_file, \
        open(benign_file, 'w', newline='') as bf, \
        open(attacks_file, 'w', newline='') as af:
    reader = csv.reader(csv_file)
    benign_writer = csv.writer(bf)
    attacks_writer = csv.writer(af)

    header = next(reader)
    label_idx = header.index('Label') if 'Label' in header else None
    attack_type_idx = header.index('Attack Type') if 'Attack Type' in header else None

    new_header = header + ['binary result', 'categorized result']
    benign_writer.writerow(new_header)
    attacks_writer.writerow(new_header)

    category_counter = 1
    seen_attack_types = {}

    for row in reader:
        row = ['-1' if v == '?' else v for v in row]
        label = row[label_idx].strip().lower() if label_idx is not None else 'benign'
        binary_result = '0' if label == 'benign' else '1'

        categorized_result = '0'
        if binary_result == '1' and attack_type_idx is not None:
            attack_type = row[attack_type_idx].strip()
            if attack_type and attack_type.lower() != 'benign':
                if attack_type not in seen_attack_types:
                    seen_attack_types[attack_type] = ((category_counter - 1) % 7) + 1
                    category_counter += 1
                categorized_result = str(seen_attack_types[attack_type])

        row_with_cols = row + [binary_result, categorized_result]
        if binary_result == '0':
            benign_writer.writerow(row_with_cols)
        else:
            attacks_writer.writerow(row_with_cols)

print("Task 1 completed: Benign.txt and Attacks.txt created.\n")

# === Task 2: Splitting Attacks.txt into categories based on 'categorized result' ===
print("Task 2: Splitting attacks into categories...")
category_files = [os.path.join(output_dir, f"Category{i}.txt") for i in range(1, 8)]
category_writers = [csv.writer(open(f, 'w', newline='')) for f in category_files]

with open(attacks_file, 'r') as af:
    reader = csv.reader(af)
    header = next(reader)
    cat_idx = header.index('categorized result')

    for w in category_writers:
        w.writerow(header)

    for row in reader:
        cat = int(row[cat_idx]) if row[cat_idx].isdigit() else 0
        if 1 <= cat <= 7:
            category_writers[cat - 1].writerow(row)

print("Task 2 completed: Category1.txt to Category7.txt created.\n")

# === Task 3: Split data into 80% train and 20% eval ===
print("Task 3: Splitting into training (80%) and evaluation (20%) sets...")
train_dir = os.path.join(output_dir, "Training_Data")
eval_dir = os.path.join(output_dir, "Evaluation_Data")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

for filename in os.listdir(output_dir):
    if filename.endswith(".txt") and filename not in ["Attacks.txt", "Benign.txt"]:
        filepath = os.path.join(output_dir, filename)
        base_name = os.path.splitext(filename)[0]

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        for suffix, data_split, out_dir in [('80', train_data, train_dir), ('20', test_data, eval_dir)]:
            with open(os.path.join(out_dir, f"{base_name}_{suffix}.txt"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data_split)

print("Task 3 completed: Training and Evaluation datasets created.\n")

# === Task 4: Sort files by 'Seq' column ===
print("Task 4: Sorting files by Seq column...")
for root, _, files in os.walk(output_dir):
    for filename in files:
        if filename.endswith(("80.txt", "20.txt")):
            filepath = os.path.join(root, filename)
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                if 'Seq' in header:
                    seq_idx = header.index('Seq')
                    rows = sorted(reader, key=lambda r: int(r[seq_idx]) if r[seq_idx].isdigit() else 0)
                else:
                    rows = list(reader)

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

print("Task 4 completed: Files sorted by Seq.\n")

# === Task 5: Merge each category file with Benign file and sort by Seq ===
print("Task 5: Merging category files with Benign files...")


def merge_and_process(directory, output_subdir):
    merged_dir = os.path.join(directory, output_subdir)
    os.makedirs(merged_dir, exist_ok=True)

    benign_file_80 = os.path.join(directory, "Benign_80.txt")
    benign_file_20 = os.path.join(directory, "Benign_20.txt")

    for suffix in ['80', '20']:
        benign_file = benign_file_80 if suffix == '80' else benign_file_20

        if not os.path.exists(benign_file):
            continue

        with open(benign_file, 'r') as f:
            reader = csv.reader(f)
            benign_header = next(reader)
            benign_data = list(reader)

        for filename in os.listdir(directory):
            if filename.endswith(f"{suffix}.txt") and not filename.startswith("Benign"):
                filepath = os.path.join(directory, filename)
                base_name = os.path.splitext(filename)[0]

                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # skip header
                    cat_data = list(reader)

                merged_file = os.path.join(merged_dir, f"{base_name}_merged.txt")
                with open(merged_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(benign_header)
                    writer.writerows(cat_data)
                    writer.writerows(benign_data)

                # Sort merged file by Seq if present
                df = pd.read_csv(merged_file, low_memory=False)
                if 'Seq' in df.columns:
                    df['Seq'] = pd.to_numeric(df['Seq'], errors='coerce')
                    df = df.sort_values('Seq')

                # Drop columns that are not needed
                for col in ['Seq', 'address', 'time']:
                    if col in df.columns:
                        df.drop(col, axis=1, inplace=True)

                final_file = merged_file.replace('_merged.txt', '_merged_final.txt')
                df.to_csv(final_file, index=False)


merge_and_process(train_dir, "Merged_Files")
merge_and_process(eval_dir, "Merged_Files")

print("Task 5 completed: Merged files created.\n")

# === Task 6 & 7: Train and evaluate SVM models ===
print("Task 6 & 7: Training and evaluating SVM models...")

balanced_accuracies = {'linear': [], 'rbf': []}
precisions = {'linear': [], 'rbf': []}
recalls = {'linear': [], 'rbf': []}
f1_scores = {'linear': [], 'rbf': []}


def train_and_evaluate_svm(category, kernel_name):
    print(f"\n{'=' * 60}")
    print(f"Category {category} - {kernel_name.upper()} Kernel")
    print(f"{'=' * 60}")

    train_file = os.path.join(train_dir, "Merged_Files", f"Category{category}_80_merged_final.txt")
    test_file = os.path.join(eval_dir, "Merged_Files", f"Category{category}_20_merged_final.txt")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Training or test file missing, skipping...")
        return

    try:
        train_data = pd.read_csv(train_file, low_memory=False)
        test_data = pd.read_csv(test_file, low_memory=False)

        exclude_cols = ['binary result', 'categorized result', 'Label', 'Attack Type', 'Attack Tool']
        feature_cols = [c for c in train_data.columns if c not in exclude_cols]

        x_train = train_data[feature_cols].copy()
        y_train = pd.to_numeric(train_data['binary result'], errors='coerce').fillna(0).astype(int)
        x_test = test_data[feature_cols].copy()
        y_test = pd.to_numeric(test_data['binary result'], errors='coerce').fillna(0).astype(int)

        # Label encode categorical features
        label_encoders = {}
        for col in x_train.columns:
            if x_train[col].dtype == 'object':
                le = LabelEncoder()
                x_train[col] = x_train[col].fillna('unknown').astype(str)
                le.fit(x_train[col].unique())
                x_train[col] = le.transform(x_train[col])
                label_encoders[col] = le

                x_test[col] = x_test[col].fillna('unknown').astype(str)
                x_test[col] = x_test[col].apply(lambda v: le.transform([v])[0] if v in le.classes_ else 0)

        # Convert to numeric
        for col in x_train.columns:
            if col not in label_encoders:
                x_train[col] = pd.to_numeric(x_train[col], errors='coerce')
                x_test[col] = pd.to_numeric(x_test[col], errors='coerce')

        x_train = x_train.fillna(0).select_dtypes(include=[np.number])
        x_test = x_test[x_train.columns].fillna(0).select_dtypes(include=[np.number])

        # Scale features
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
        x_test_scaled = scaler.transform(x_test.astype(np.float32))

        # Train SVM
        print(f"Training on {len(x_train_scaled)} samples...")
        start_time = time.time()

        if kernel_name == 'linear':
            model = LinearSVC(C=1.0, max_iter=10000, dual=False, tol=1e-4, random_state=42)
        else:
            model = SVC(kernel=kernel_name, C=1.0, max_iter=10000, tol=1e-4, random_state=42)

        model.fit(x_train_scaled, y_train)
        print(f"Training time: {time.time() - start_time:.2f}s")

        # Predict and evaluate
        y_pred = model.predict(x_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)

        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
        balanced_acc = (TPR + TNR) / 2

        balanced_accuracies[kernel_name].append((category, balanced_acc))
        precisions[kernel_name].append((category, report['macro avg']['precision']))
        recalls[kernel_name].append((category, report['macro avg']['recall']))
        f1_scores[kernel_name].append((category, report['macro avg']['f1-score']))

        # Save model
        model_dir = os.path.join(BASE_DIR, 'SVM_Models', kernel_name.capitalize() + '_Kernel')
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, f'svm_category{category}_{kernel_name}.pkl'))

        # Print metrics
        print(f"\nConfusion Matrix:\nTN: {TN}, FP: {FP}\nFN: {FN}, TP: {TP}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=1)}")

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix - Category {category} - {kernel_name.capitalize()} Kernel")
        plt.savefig(os.path.join(model_dir, f'cm_category{category}_{kernel_name}.png'))
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Error: {str(e)}")


# Run training and evaluation
kernels = ['linear', 'rbf']
categories = [1, 2, 3]

start_total = time.time()
for cat in categories:
    for kernel in kernels:
        train_and_evaluate_svm(cat, kernel)

print(f"\nAll models completed in {(time.time() - start_total) / 60:.2f} minutes\n")


# === Plot comparison charts ===
def plot_metric_comparison(metric_dict, metric_name):
    categories_sorted = sorted(set(cat for cat, _ in metric_dict['linear']))
    linear_scores = [score for cat, score in sorted(metric_dict['linear'])]
    rbf_scores = [score for cat, score in sorted(metric_dict['rbf'])]
    x = range(len(categories_sorted))

    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], linear_scores, width=0.4, label='Linear Kernel')
    plt.bar([i + 0.2 for i in x], rbf_scores, width=0.4, label='RBF Kernel')
    plt.xticks(x, [f'Category {c}' for c in categories_sorted])
    plt.ylim(0, 1.05)
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Comparison: Linear vs RBF Kernels')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


plot_metric_comparison(balanced_accuracies, "Balanced Accuracy")
plot_metric_comparison(precisions, "Precision")
plot_metric_comparison(recalls, "Recall")
plot_metric_comparison(f1_scores, "F1-Score")
