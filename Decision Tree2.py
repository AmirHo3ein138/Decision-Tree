import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from scipy.stats import f_oneway
import matplotlib.pyplot as plt

# Read the sample data and define the label column
df_sample = pd.read_csv("cicids2017_sampled.csv")
label_column = 'Attack Type'

# Separate features and labels
X = df_sample.drop(label_column, axis=1)
y = df_sample[label_column]

# Store initial features
initial_features = df_sample.drop(label_column, axis=1).columns.tolist()

# Remove correlated features (with a threshold of 0.95)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X = X.drop(columns=to_drop)

def compare_features(initial_features, final_features):
    with open('final_features.txt', 'a', encoding='utf-8') as f:
        f.write("\n--- Comparison of Initial and Final Features ---\n")
        f.write("\nInitial Features (before processing):\n")
        for feature in initial_features:
            f.write(f"- {feature}\n")
        
        f.write("\nFinal Features (after removing correlated ones):\n")
        for feature in final_features:
            f.write(f"- {feature}\n")
        
        removed_features = [f for f in initial_features if f not in final_features]
        if removed_features:
            f.write("\nRemoved Features:\n")
            for feature in removed_features:
                f.write(f"- {feature}\n")
        else:
            f.write("\nNo features were removed.\n")

compare_features(initial_features, X.columns.tolist())

# Discretize numerical features
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        discretizer = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='kmeans')
        X[col] = discretizer.fit_transform(X[[col]]).astype(int)

# Split data into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

# Define Leaf Node class
class LeafNode:
    def __init__(self, labels):
        self.prediction = Counter(labels).most_common(1)[0][0]
    def is_leaf(self):
        return True

# Define Decision Node class
class DecisionNode:
    def __init__(self, feature, children):
        self.feature = feature
        self.children = children
    def is_leaf(self):
        return False

# Calculate entropy
def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * np.log2(p) if p > 0 else 0
    return ent

# Calculate Information Gain
def information_gain(y, x_column):
    parent_entropy = entropy(y)
    values, counts = np.unique(x_column, return_counts=True)
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * entropy(y[x_column == values[i]])
        for i in range(len(values))
    ])
    return parent_entropy - weighted_entropy

# Calculate Gini Index
def gini_index(labels):
    counts = Counter(labels)
    total = len(labels)
    gini = 1.0 - sum((count / total) ** 2 for count in counts.values())
    return gini

# Calculate Gini Gain
def gini_gain(y, x_column):
    parent_gini = gini_index(y)
    values, counts = np.unique(x_column, return_counts=True)
    weighted_gini = np.sum([
        (counts[i] / np.sum(counts)) * gini_index(y[x_column == values[i]])
        for i in range(len(values))
    ])
    return parent_gini - weighted_gini

# Build decision tree recursively with criterion option
def build_tree(X, y, depth=0, max_depth=7, min_samples_split=30, min_gain=0.01, criterion='ig'):
    if len(set(y)) == 1 or depth == max_depth or len(y) < min_samples_split:
        return LeafNode(y)

    best_gain = -1
    best_feature = None
    for feature in X.columns:
        if criterion == 'ig':
            gain = information_gain(y, X[feature])
        else:
            gain = gini_gain(y, X[feature])
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    if best_gain <= min_gain or best_feature is None:
        return LeafNode(y)

    children = {}
    for val in X[best_feature].unique():
        idx = X[best_feature] == val
        child = build_tree(X[idx], y[idx], depth+1, max_depth, min_samples_split, min_gain, criterion)
        children[val] = child

    return DecisionNode(best_feature, children)

# Predict for a single sample
def predict_single(x, node, default):
    while not node.is_leaf():
        val = x[node.feature]
        if val in node.children:
            node = node.children[val]
        else:
            return default
    return node.prediction

# Predict for all test data
def predict(X, tree, default):
    return [predict_single(row, tree, default) for _, row in X.iterrows()]

# Find majority class
majority_class = y_train.mode()[0]

# Pruning function with validation set
def prune_tree(node, X_val, y_val):
    if node.is_leaf():
        return node

    for val, child in node.children.items():
        idx = X_val[node.feature] == val
        node.children[val] = prune_tree(child, X_val[idx], y_val[idx])

    preds_before = [predict_single(row, node, majority_class) for _, row in X_val.iterrows()]
    acc_before = accuracy_score(y_val, preds_before)

    majority = Counter(y_val).most_common(1)[0][0]
    acc_leaf = np.mean([majority == true for true in y_val])

    # Prune if leaf accuracy is higher
    if acc_leaf > acc_before:
        return LeafNode(y_val)
    return node

# Build tree with Information Gain
print("Building tree with Information Gain")
tree_ig = build_tree(X_train, y_train, max_depth=7, min_samples_split=30, min_gain=0.01, criterion='ig')
tree_ig = prune_tree(tree_ig, X_val, y_val)
y_pred_ig = predict(X_test, tree_ig, majority_class)
print(classification_report(y_test, y_pred_ig))

# Build tree with Gini Index
print("Building tree with Gini Index")
tree_gini = build_tree(X_train, y_train, max_depth=7, min_samples_split=30, min_gain=0.01, criterion='gini')
tree_gini = prune_tree(tree_gini, X_val, y_val)
y_pred_gini = predict(X_test, tree_gini, majority_class)
print(classification_report(y_test, y_pred_gini))

# Plot tree with IG and Gini display
def plot_tree(node, ax, x, y, x_offset=2.0, level=0, X=None, labels=None, criterion='ig'):
    if node.is_leaf():
        ax.text(x, y, f"Leaf: {node.prediction}", ha='center', va='center', bbox=dict(facecolor='lightgreen', alpha=0.5), fontsize=8)
        return x + x_offset, [x]
    ig = information_gain(labels, X[node.feature]) if X is not None and labels is not None else 0
    gini_val = gini_gain(labels, X[node.feature]) if X is not None and labels is not None else 0
    ax.text(x, y, f"{node.feature}\nIG: {ig:.2f}\nGini Gain: {gini_val:.2f}", ha='center', va='center', bbox=dict(facecolor='lightblue', alpha=0.5), fontsize=8)
    child_x = x
    child_positions = [child_x]
    for val, child in node.children.items():
        idx = X[node.feature] == val if X is not None else None
        child_X = X[idx] if idx is not None else None
        child_labels = labels[idx] if idx is not None else None
        child_x, positions = plot_tree(child, ax, child_x + x_offset, y - 1, x_offset, level + 1, child_X, child_labels, criterion)
        ax.plot([x, child_x], [y - 0.1, y - 0.9], 'k-')
        ax.text((x + child_x) / 2, y - 0.5, str(val), ha='center', va='center', fontsize=6)
        child_positions.extend(positions)
    return child_x, child_positions

# Plot tree with IG
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('Decision Tree Visualization (with IG and Gini)')
ax.axis('off')
plot_tree(tree_ig, ax, 0, 7, x_offset=1.5, X=X_train, labels=y_train, criterion='ig')
plt.show()

# Calculate and display tree statistics
def tree_stats(node, depth=0):
    if node.is_leaf():
        return {'leaves': 1, 'nodes': 1, 'max_depth': depth}
    stats = {'leaves': 0, 'nodes': 1, 'max_depth': depth}
    for child in node.children.values():
        child_stats = tree_stats(child, depth+1)
        stats['leaves'] += child_stats['leaves']
        stats['nodes'] += child_stats['nodes']
        stats['max_depth'] = max(stats['max_depth'], child_stats['max_depth'])
    return stats

stats = tree_stats(tree_ig)
accuracy = accuracy_score(y_test, y_pred_ig)
print(f"Accuracy (IG): {accuracy:.2f}, Leaves: {stats['leaves']}, Nodes: {stats['nodes']}, Max Depth: {stats['max_depth']}")

# ---------------------- Advanced statistical analysis to txt file ----------------------

with open('advanced_stats.txt', 'w', encoding='utf-8') as f:
    f.write("--- EDA: Descriptive Statistics ---\n")
    f.write(df_sample.describe().to_string())
    f.write("\n\n--- Correlation Matrix ---\n")
    numeric_df = df_sample.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    f.write(corr_matrix.to_string())
    
    f.write("\n\n--- Feature Importance based on Correlation ---\n")
    le = LabelEncoder()
    df_sample['Attack Type Enc'] = le.fit_transform(df_sample[label_column])
    correlations = numeric_df.corrwith(df_sample['Attack Type Enc'])
    f.write(correlations.sort_values(ascending=False).to_string())
    
    f.write("\n\n--- ANOVA Test for Flow Duration ---\n")
    groups = [df_sample[df_sample[label_column] == label]['Flow Duration'] for label in df_sample[label_column].unique()]
    anova_result = f_oneway(*groups)
    f.write(f"F-stat: {anova_result.statistic:.2f}, p-value: {anova_result.pvalue:.4f}\n")
    
    f.write("\n--- Confusion Matrix (saved as image) ---\n")
    cm = confusion_matrix(y_test, y_pred_ig)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    f.write("Confusion Matrix saved to file confusion_matrix.png.\n")
    
    f.write("\n--- Cross-Validation Accuracy ---\n")
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=7, min_samples_split=30, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    f.write(f"Mean: {cv_scores.mean():.2f} ± Std: {cv_scores.std():.2f}\n")
    
    f.write("\n--- Feature Importance in Tree (with cumulative gain) ---\n")
    def feature_importance(node, counter, X, y, depth=0):
        if node.is_leaf():
            return
        counter[node.feature] += information_gain(y, X[node.feature]) / (depth + 1)
        for val, child in node.children.items():
            idx = X[node.feature] == val
            feature_importance(child, counter, X[idx], y[idx], depth + 1)
    fi_counter = Counter()
    feature_importance(tree_ig, fi_counter, X_train, y_train)
    fi_df = pd.DataFrame({'Feature': list(fi_counter.keys()), 'Importance': list(fi_counter.values())}).sort_values('Importance', ascending=False)
    f.write(fi_df.to_string(index=False))

print("Advanced statistical analysis saved to advanced_stats.txt.")