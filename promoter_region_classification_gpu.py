import pandas as pd
import numpy as np


####################### PHASE 1 #############################
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'
columns = ['Class', 'id', 'Sequence']
genes = pd.read_csv(url, names=columns)

genes['Sequence'] = genes['Sequence'].str.strip()   # strip whitespace/tabs
genes['Class'] = genes['Class'].map({'+': 1, '-': 0})  # promoter=1, non-promoter=0

print(genes.shape)   #(106, 3)
print(genes.head())
print(genes['Sequence'].str.len().describe())
print(genes['Sequence'].apply(lambda s: s != s.strip()).sum())  # any leading/trailing whitespace?
print(set(''.join(genes['Sequence'])))  # every unique character across all sequences


###################### Base Map Indexing and Encoding ######################################
base_to_idx = {'a' : 0, 'c' : 1, 't':2, 'g':3}

encoded = np.zeros((57,4), dtype = np.float32) #Creates the 57x4 array with 0s to start
print()
print(encoded.shape)
print(encoded)
print()


def one_hot_encode(seq):
    encoded = np.zeros((len(seq),4), dtype = np.float32)

    for i, base in enumerate(seq):
        encoded[i, base_to_idx[base]] = 1

    return encoded

print()
print('################# Test Ecoded Shape #################')
test = one_hot_encode(genes['Sequence'][0])
print(test.shape)   # expect (57, 4)
print(test.sum())   # expect 57.0 — one '1' per row, 57 rows

results = []
for seq in genes['Sequence']:
    results.append(one_hot_encode(seq))
print()
print('################# Gene Ecoded Shape #################\n', len(results)) #expect 106
print(results[0].shape)     # expect (57, 4)
X = np.stack(results)
print()
print(X.shape)

y= genes['Class'].values.astype(np.float32)
print(y.shape)
print(y[:10]) #First 5 labels

####################### PHASE 2 #############################

################################# Train/Val Split


from sklearn.model_selection import StratifiedKFold
import torch 
import torch.nn as nn

class PromoterBaseline(nn.Module):
    def __init__(self, in_features):      # "in_features" here is just a name/placeholder
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
X_tensor = torch.tensor(X.reshape(106,228)).to(device)
y_tensor = torch.tensor(y).to(device)
print()
print('################# Train and Value Shapes #################\n')
fold_accuracies = []
for train_idx, val_idx in skf.split(X,y):
    X_train = X_tensor[train_idx]
    print(X_train.shape)
    y_train = y_tensor[train_idx]
    print(y_train.shape)
    X_val = X_tensor[val_idx]
    print(X_val.shape)
    y_val = y_tensor[val_idx]
    print(y_val.shape)

    model = PromoterBaseline(X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_probs = torch.sigmoid(val_outputs)
        val_preds = (val_probs > 0.5).float()     # threshold val_probs at 0.5 to get 0/1 predictions
        accuracy = (val_preds==y_val).float().mean()   # compare val_preds to y_val, get the fraction correct
        fold_accuracies.append(accuracy.item())
    print(f"Fold val accuracy: {accuracy}")
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)
print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

######### CNN
X_cnn = np.transpose(X, (0,2,1))
X_cnn_tensor = torch.tensor(X_cnn).to(device)

class PromoterCNN(nn.Module):
    def __init__(self, seq_len, n_filters=32, kernel_size=8):
        super().__init__()
        self.conv1 = nn.Conv1d(4, n_filters, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)
        
torch.manual_seed(42)
cnn_fold_accuracies = []
print()
print('################# CNN Train and Value Shapes #################\n')
for train_idx, val_idx in skf.split(X, y):
    X_train = X_cnn_tensor[train_idx]
    y_train = y_tensor[train_idx]
    X_val = X_cnn_tensor[val_idx]
    y_val = y_tensor[val_idx]

    model = PromoterCNN(seq_len=57).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_outputs = model(X_val)
        val_probs = torch.sigmoid(val_outputs)
        val_preds = (val_probs > 0.5).float()
        accuracy = (val_preds == y_val).float().mean()
        cnn_fold_accuracies.append(accuracy.item())
    print(f"Fold val accuracy: {accuracy}")

mean_acc = np.mean(cnn_fold_accuracies)
std_acc = np.std(cnn_fold_accuracies)
print(f"CNN Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

################### PHASE 3 ############################
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

accuracies = []
precisions = []
recalls = []
f1s = []
aucs = []
confusion_matrices = []

torch.manual_seed(42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train = X_cnn_tensor[train_idx]
    y_train = y_tensor[train_idx]
    X_val = X_cnn_tensor[val_idx]
    y_val = y_tensor[val_idx]

    model = PromoterCNN(seq_len=57).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs >= 0.5).float()

    y_val_np = y_val.cpu().numpy()
    val_preds_np = val_preds.cpu().numpy()
    val_probs_np = val_probs.cpu().numpy()

    accuracy = (val_preds_np == y_val_np).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val_np, val_preds_np, average='binary'
    )
    auc = roc_auc_score(y_val_np, val_probs_np)
    cm = confusion_matrix(y_val_np, val_preds_np)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    aucs.append(auc)
    confusion_matrices.append(cm)

print(f"Accuracy:  {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}")
print(f"Recall:    {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")
print(f"F1:        {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
print(f"ROC-AUC:   {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

################### BASELINE ############################
baseline_accuracies = []
baseline_precisions = []
baseline_recalls = []
baseline_f1s = []
baseline_aucs = []
baseline_confusion_matrices = []

torch.manual_seed(42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print()
print('################# Baseline #################\n')



for train_idx, val_idx in skf.split(X, y):
    X_train = X_tensor[train_idx]
    y_train = y_tensor[train_idx]
    X_val = X_tensor[val_idx]
    y_val = y_tensor[val_idx]

    model = PromoterBaseline(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs >= 0.5).float()

    y_val_np = y_val.cpu().numpy()
    val_preds_np = val_preds.cpu().numpy()
    val_probs_np = val_probs.cpu().numpy()

    accuracy = (val_preds_np == y_val_np).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val_np, val_preds_np, average='binary'
    )
    auc = roc_auc_score(y_val_np, val_probs_np)
    cm = confusion_matrix(y_val_np, val_preds_np)

    baseline_accuracies.append(accuracy)
    baseline_precisions.append(precision)
    baseline_recalls.append(recall)
    baseline_f1s.append(f1)
    baseline_aucs.append(auc)
    baseline_confusion_matrices.append(cm)

print(f"Baseline Accuracy:  {np.mean(baseline_accuracies):.4f} +/- {np.std(baseline_accuracies):.4f}")
print(f"Baseline Precision: {np.mean(baseline_precisions):.4f} +/- {np.std(baseline_precisions):.4f}")
print(f"Baseline Recall:    {np.mean(baseline_recalls):.4f} +/- {np.std(baseline_recalls):.4f}")
print(f"Baseline F1:        {np.mean(baseline_f1s):.4f} +/- {np.std(baseline_f1s):.4f}")
print(f"Baseline ROC-AUC:   {np.mean(baseline_aucs):.4f} +/- {np.std(baseline_aucs):.4f}")

################### CONFUSION MATRIX AGGREGATION ############################

# CNN
cnn_cm_total = np.sum(confusion_matrices, axis = 0)
tn, fp, fn, tp = cnn_cm_total.ravel()

print()
print()
print("CNN aggregated confusion matrix:")
print(cnn_cm_total)
print(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")

# Baseline
baseline_cm_total = np.sum(baseline_confusion_matrices, axis = 0)
tn_b, fp_b, fn_b, tp_b = baseline_cm_total.ravel()

print("Baseline aggregated confusion matrix:")
print(baseline_cm_total)
print(f"TN: {tn_b}  FP: {fp_b}  FN: {fn_b}  TP: {tp_b}")

################### FINAL CNN (trained on full dataset, for interpretation) ############################

torch.manual_seed(42)
final_cnn = PromoterCNN(seq_len=57).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(final_cnn.parameters(), lr=1e-3)

num_epochs = 100
for epoch in range(num_epochs):
    final_cnn.train()
    optimizer.zero_grad()
    logits = final_cnn(X_cnn_tensor)
    loss = criterion(logits, y_tensor)
    loss.backward()
    optimizer.step()

final_cnn.eval()
with torch.no_grad():
    final_logits = final_cnn(X_cnn_tensor)
    final_probs = torch.sigmoid(final_logits)
    final_preds = (final_probs >= 0.5).float()
    final_train_accuracy = (final_preds == y_tensor).float().mean()

print(f"Final CNN (trained on full dataset) training accuracy: {final_train_accuracy.item():.4f}")


################### MOTIF VISUALIZATION ############################

with torch.no_grad():
    conv_out = final_cnn.conv1(X_cnn_tensor)   # shape: (106, 32, 50)

kernel_size = 8
num_positions = conv_out.shape[2]   # 50

print("Top motif per CNN filter:\n")
for filter_idx in range(conv_out.shape[1]):
    filter_activations = conv_out[:, filter_idx, :]        # shape: (106, 50)
    flat_idx = torch.argmax(filter_activations).item()     # single flat index into the (106*50) grid
    seq_idx, position_idx = divmod(flat_idx, num_positions)

    motif = genes['Sequence'].iloc[seq_idx][position_idx : position_idx + kernel_size]
    activation_value = filter_activations[seq_idx, position_idx].item()

    print(f"Filter {filter_idx:2d}: {motif}  (activation={activation_value:.3f}, seq={seq_idx}, pos={position_idx})")

################### FILTER POSITION CLUSTERING ############################
import matplotlib.pyplot as plt

positions = []
for filter_idx in range(conv_out.shape[1]):
    filter_activations = conv_out[:, filter_idx, :]
    flat_idx = torch.argmax(filter_activations).item()
    seq_idx, position_idx = divmod(flat_idx, num_positions)
    activation_value = filter_activations[seq_idx, position_idx].item()

    if activation_value > 0:   # skip dead filters like filter 3
        positions.append(position_idx)

print(f"Positions used (excluding dead filters): {positions}")

plt.figure(figsize=(8, 4))
plt.hist(positions, bins=range(0, num_positions + 1), color="#3B6FA0", edgecolor="white")
plt.xlabel("Position in sequence (0 = start of 57-base window)")
plt.ylabel("Number of filters peaking here")
plt.title("Where CNN filters' top motifs occur along the sequence")
plt.tight_layout()
plt.savefig("filter_position_histogram.png")
plt.show()

################### PHASE 4 ############################
import random

def mutate_sequence(seq, mutation_rate):
    bases = ['a', 'c', 'g', 't']
    mutated = list(seq)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            other_bases = [b for b in bases if b != mutated[i]]
            mutated[i] = random.choice(other_bases)
    return ''.join(mutated)

################### RETRAIN & STORE FOLD MODELS ############################

torch.manual_seed(42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_models = []       # will hold (model, val_idx) pairs

for train_idx, val_idx in skf.split(X, y):
    X_train = X_cnn_tensor[train_idx]
    y_train = y_tensor[train_idx]

    model = PromoterCNN(seq_len=57).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    fold_models.append((model, val_idx))

print(f"Stored {len(fold_models)} trained fold models.")

################### MUTATION RATE STRESS TEST ############################

mutation_rates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
sequences = genes['Sequence'].values   # raw sequences, same order as X/y

rate_mean_accuracies = []
rate_std_accuracies = []

for rate in mutation_rates:
    fold_accuracies = []

    for model, val_idx in fold_models:
        val_sequences = sequences[val_idx]
        y_val = y[val_idx]

        mutated_sequences = [mutate_sequence(seq, rate) for seq in val_sequences]

        encoded = np.stack([one_hot_encode(seq) for seq in mutated_sequences])   # (n_val, 57, 4)
        encoded_cnn = np.transpose(encoded, (0, 2, 1))                           # (n_val, 4, 57)
        X_val_mutated = torch.tensor(encoded_cnn, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(X_val_mutated)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            accuracy = (preds == y_val_tensor).float().mean().item()

        fold_accuracies.append(accuracy)

    rate_mean_accuracies.append(np.mean(fold_accuracies))
    rate_std_accuracies.append(np.std(fold_accuracies))
    print(f"Mutation rate {rate:.2f}: accuracy {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")

plt.figure(figsize=(8, 5))
plt.errorbar(mutation_rates, rate_mean_accuracies, yerr=rate_std_accuracies,
             color = "#3B6FA0", marker="o", linewidth=2, capsize=4)
plt.xlabel("Mutation rate")
plt.ylabel("Validation accuracy")
plt.title("CNN accuracy degradation under synthetic point mutations")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("mutation_stress_test.png")
plt.show()