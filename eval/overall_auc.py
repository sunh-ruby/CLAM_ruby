import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_prob_label(data):
    probs = []
    labels = []
    for key in data:
        probs.append(data[key]['prob'][0][1])  # Assuming the second column is the positive class probability
        labels.append(data[key]['label'])
    return np.array(probs), np.array(labels)

def plot_roc_curves(input_dir="/home/harry/Documents/codes/CLAM_ruby/results/KRAS_ruby_s1"):
    all_fprs = []
    all_tprs = []
    aucs = []
    curves = []

    plt.figure(figsize=(10, 8))

    for file in os.listdir(input_dir):
        if file.startswith("split_") and file.endswith("_results.pkl"):
            file_path = os.path.join(input_dir, file)
            data = load_pkl(file_path)
            probs, labels = extract_prob_label(data)
            
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            curves.append((fpr, tpr, roc_auc))
            
            # Interpolate TPR values to a common set of FPR points
            interp_tpr = np.interp(np.linspace(0, 1, 100), fpr, tpr)
            interp_tpr[0] = 0.0
            all_tprs.append(interp_tpr)
            all_fprs = np.linspace(0, 1, 100)

    # Calculate median and std of TPRs
    tprs = np.array(all_tprs)
    mean_tprs = np.mean(tprs, axis=0)
    std_tprs = np.std(tprs, axis=0)

    # Calculate median and max AUC
    median_auc = np.median(aucs)
    max_auc = np.max(aucs)
    mean_auc = np.mean(aucs)

    # Plot the median ROC curve with std shading
    plt.plot(all_fprs, mean_tprs, color='blue', label=f'Median ROC (AUC = {median_auc:.2f})')
    plt.fill_between(all_fprs, mean_tprs - std_tprs, mean_tprs + std_tprs, color='gray', alpha=0.3, label=f'± 1 std. dev.')

    # Plot the highest AUC curve
    max_auc_index = np.argmax(aucs)
    fpr_max, tpr_max, _ = curves[max_auc_index]
    plt.plot(fpr_max, tpr_max, color='pink', linestyle='-', linewidth=2, label=f'Max ROC (AUC = {max_auc:.2f})')

    # Plot settings
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves')
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig('roc_curves_with_max_highlighted.png')
    plt.close()

    print(f"Median AUC: {median_auc:.2f}")
    print(f"Mean AUC: {mean_auc:.2f}")
    print(f"Max AUC: {max_auc:.2f}")

if __name__ == "__main__":
    plot_roc_curves()