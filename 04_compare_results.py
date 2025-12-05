import pandas as pd
import os

# ----------------------------
# Configuration
# ----------------------------
original_summary_file = "results/original_summary_report.csv"
masked_summary_file = "results/masked_summary_report.csv"
original_detailed_file = "results/original_detailed_report.csv"
masked_detailed_file = "results/masked_detailed_report.csv"
comparison_file = "results/comparison_summary.csv"

# ----------------------------
# Helper: Compute class-wise top5 accuracy from detailed CSV
# ----------------------------
def compute_class_top5(detailed_csv):
    df = pd.read_csv(detailed_csv)
    classes = df['true_label'].unique()
    top5_dict = {}
    for cls in classes:
        cls_df = df[df['true_label'] == cls]
        correct_top5 = cls_df.apply(lambda row: row['true_label'] in
                                    [row[f"top{i}_class"] for i in range(1,6)], axis=1).sum()
        total = len(cls_df)
        top5_dict[cls] = correct_top5 / total * 100
    return top5_dict

# ----------------------------
# Load summary CSVs for Top-1
# ----------------------------
cols = ['Class', 'Correct', 'Total', 'Top1_Accuracy']
df_orig = pd.read_csv(original_summary_file, header=None, names=cols)
df_masked = pd.read_csv(masked_summary_file, header=None, names=cols)

class_df_orig = df_orig[~df_orig['Class'].str.contains('Overall', na=False)].copy()
class_df_masked = df_masked[~df_masked['Class'].str.contains('Overall', na=False)].copy()

class_df_orig['Top1_Accuracy'] = pd.to_numeric(class_df_orig['Top1_Accuracy'], errors='coerce')
class_df_masked['Top1_Accuracy'] = pd.to_numeric(class_df_masked['Top1_Accuracy'], errors='coerce')

# ----------------------------
# Load Top-5 per class from detailed CSVs
# ----------------------------
orig_top5_dict = compute_class_top5(original_detailed_file)
masked_top5_dict = compute_class_top5(masked_detailed_file)

# ----------------------------
# Merge into comparison dataframe
# ----------------------------
comparison_df = pd.DataFrame()
comparison_df['Class'] = class_df_orig['Class']
comparison_df['Original_Top1'] = class_df_orig['Top1_Accuracy']
comparison_df['Masked_Top1'] = class_df_masked['Top1_Accuracy']
comparison_df['Delta_Top1'] = comparison_df['Masked_Top1'] - comparison_df['Original_Top1']

comparison_df['Original_Top5'] = comparison_df['Class'].map(orig_top5_dict)
comparison_df['Masked_Top5'] = comparison_df['Class'].map(masked_top5_dict)
comparison_df['Delta_Top5'] = comparison_df['Masked_Top5'] - comparison_df['Original_Top5']

# ----------------------------
# Compute overall Top-1 and Top-5
# ----------------------------
overall_orig_top1 = pd.to_numeric(df_orig[df_orig['Class'].str.contains('Overall Top-1')]['Top1_Accuracy'].iloc[0])
overall_masked_top1 = pd.to_numeric(df_masked[df_masked['Class'].str.contains('Overall Top-1')]['Top1_Accuracy'].iloc[0])
delta_overall_top1 = overall_masked_top1 - overall_orig_top1

overall_orig_top5 = pd.to_numeric(df_orig[df_orig['Class'].str.contains('Overall Top-5')]['Top1_Accuracy'].iloc[0])
overall_masked_top5 = pd.to_numeric(df_masked[df_masked['Class'].str.contains('Overall Top-5')]['Top1_Accuracy'].iloc[0])
delta_overall_top5 = overall_masked_top5 - overall_orig_top5

# ----------------------------
# Print comparison in logs
# ----------------------------
print("\n📊 MODEL COMPARISON SUMMARY (Top-1 & Top-5)")
print("="*100)
print(f"{'Class':<25} {'Orig Top1':>10} {'Masked Top1':>12} {'Δ Top1':>10} "
      f"{'Orig Top5':>12} {'Masked Top5':>12} {'Δ Top5':>10}")
print("-"*100)
for _, row in comparison_df.iterrows():
    print(f"{row['Class']:<25} "
          f"{row['Original_Top1']:>9.2f}% {row['Masked_Top1']:>11.2f}% {row['Delta_Top1']:>9.2f}% "
          f"{row['Original_Top5']:>11.2f}% {row['Masked_Top5']:>11.2f}% {row['Delta_Top5']:>9.2f}%")
print("-"*100)
print(f"{'Overall Top-1 Accuracy':<25} {overall_orig_top1:>9.2f}% {overall_masked_top1:>11.2f}% {delta_overall_top1:>9.2f}%")
print(f"{'Overall Top-5 Accuracy':<25} {overall_orig_top5:>9.2f}% {overall_masked_top5:>11.2f}% {delta_overall_top5:>9.2f}%")
print("="*100)

# ----------------------------
# Save class-wise comparison CSV
# ----------------------------
os.makedirs("results", exist_ok=True)
comparison_df.to_csv(comparison_file, index=False)
print(f"\n✅ Comparison CSV saved to {comparison_file}")
