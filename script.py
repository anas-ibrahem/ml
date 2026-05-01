import pandas as pd
import numpy as np

# Load true labels
true_df = pd.read_csv('true_solutions.csv')
y_true = true_df['True_Label'].values

# Load predicted labels
with open('results.txt', 'r') as f:
    y_pred = [int(line.strip()) for line in f if line.strip()]
y_pred = np.array(y_pred)

# Compare lengths
if len(y_true) != len(y_pred):
    print(f'Length mismatch! True: {len(y_true)}, Pred: {len(y_pred)}')
else:
    # Calculate accuracy
    acc = np.mean(y_true == y_pred)
    print(f'Accuracy: {acc * 100:.2f}%')
    
    # Calculate per-class accuracy
    from sklearn.metrics import classification_report
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred))