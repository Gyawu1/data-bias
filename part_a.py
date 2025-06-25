#names groq
import pandas as pd

# Loading both CSVs
black_df = pd.read_csv('/content/black_individuals_metadata.csv')
white_df = pd.read_csv('/content/white_individuals_metadata.csv')

# Adding race label
black_df['race'] = 'Black'
white_df['race'] = 'White'

# Combining into one DataFrame
df = pd.concat([black_df, white_df], ignore_index=True)

# Confirming structure
print(df.head())
print(df['race'].value_counts())
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x='race', palette='coolwarm')
plt.title("Image Count by Racial Group")
plt.xlabel("Race")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()
df['name_length'] = df['individual_name'].apply(len)

sns.boxplot(data=df, x='race', y='name_length', palette='coolwarm')
plt.title("Name Length Distribution by Race")
plt.xlabel("Race")
plt.ylabel("Character Count")
plt.tight_layout()
plt.show()
import pandas as pd

# Loading both CSVs
black_df = pd.read_csv('/content/black_individuals_metadata.csv')
white_df = pd.read_csv('/content/white_individuals_metadata.csv')

# Adding race label
black_df['race'] = 'Black'
white_df['race'] = 'White'

# Combing into one DataFrame
df = pd.concat([black_df, white_df], ignore_index=True)

# Confirming structure
print(df.head())
print(df['race'].value_counts())

from PIL import Image
import matplotlib.pyplot as plt


def show_images(df, group, n=5):
    subset = df[df['race'] == group].sample(n)
    fig, axes = plt.subplots(1, n, figsize=(15, 4))
    for i, row in enumerate(subset.iterrows()):
        # Use the image_path from the DataFrame
        try:
            img = Image.open(row[1]['image_path'])
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(group)
        except FileNotFoundError:
            print(f"Error: Image file not found at {row[1]['image_path']}")
            # Optionally, you could display a placeholder or skip this image
            axes[i].set_title(f"Error: {group}")
            axes[i].axis('off')
    plt.suptitle(f"{group} Group Samples")
    plt.tight_layout()
    plt.show()

# Show Black samples
show_images(df, 'Black')

# Show White samples
show_images(df, 'White')
import os
print(os.getcwd())
def get_dimensions(path):
    img = Image.open(path)
    return img.size

df[['width', 'height']] = df['file_/content/img1.jpg'].apply(get_dimensions).apply(pd.Series)

# Compare resolution per group
sns.boxplot(x='race', y='width', data=df)
plt.title("Image Width by Racial Group")
plt.show()

sns.boxplot(x='race', y='height', data=df)
plt.title("Image Height by Racial Group")
plt.show()
from PIL import Image

def get_image_dimensions(df):
    widths, heights, group_labels = [], [], []
    for _, row in df.iterrows():
        try:
            img = Image.open(row['image_path'])
            width, height = img.size
            widths.append(width)
            heights.append(height)
            group_labels.append(row['race'])
        except Exception as e:
            print(f"Error loading {row['image_path']}: {e}")
    return pd.DataFrame({
        "width": widths,
        "height": heights,
        "race": group_labels
    })

# Run this
dims_df = get_image_dimensions(df)
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=dims_df, x="race", y="width")
plt.title("Width Distribution by Race")
plt.show()

sns.boxplot(data=dims_df, x="race", y="height")
plt.title("Height Distribution by Race")
plt.show()
def show_grid(df, race, n=12):
    sample = df[df['race'] == race].sample(n)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i, row in enumerate(sample.iterrows()):
        img = Image.open(row[1]['image_path'])
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(f"Random Sample Grid - {race}", fontsize=16)
    plt.tight_layout()
    plt.show()

show_grid(df, "Black")
show_grid(df, "White")


import face_recognition
import numpy as np
from tqdm import tqdm

def extract_embeddings(df):
    embeddings, races = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img = face_recognition.load_image_file(row['image_path'])
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                embeddings.append(face_encodings[0])
                races.append(row['race'])
        except Exception as e:
            print(f"Failed to process {row['image_path']}: {e}")
    return pd.DataFrame(embeddings), races

# Run this to extract
embeddings_df, race_labels = extract_embeddings(df)
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings_df)

viz_df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
viz_df['race'] = race_labels

plt.figure(figsize=(10, 7))
sns.scatterplot(data=viz_df, x='PC1', y='PC2', hue='race', alpha=0.7)
plt.title('Embedding Space by Race (PCA Projection)')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Add race labels
embeddings_df['race'] = race_labels

# Convert race to numeric
embeddings_df['label'] = embeddings_df['race'].map({'Black': 0, 'White': 1})

X = embeddings_df.drop(['race', 'label'], axis=1)
y = embeddings_df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Overall Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Black', 'White']))
# Label & Split
embeddings_df_new['race'] = race_labels_new
embeddings_df_new['label'] = embeddings_df_new['race'].map({'Black': 0, 'White': 1})

X_new = embeddings_df_new.drop(['race', 'label'], axis=1)
y_new = embeddings_df_new['label']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, stratify=y_new, test_size=0.3, random_state=42)

# Train
clf_new = LogisticRegression(max_iter=1000)
clf_new.fit(X_train_new, y_train_new)
y_pred_new = clf_new.predict(X_test_new)

# Evaluate
print("Mitigated Accuracy:", accuracy_score(y_test_new, y_pred_new))
print(classification_report(y_test_new, y_pred_new, target_names=['Black', 'White']))
from sklearn.metrics import confusion_matrix

def group_fairness_report(y_true, y_pred, group_ids, group_name):
    indices = [i for i, g in enumerate(group_ids) if g == group_name]
    y_true_group = [y_true[i] for i in indices]
    y_pred_group = [y_pred[i] for i in indices]
    cm = confusion_matrix(y_true_group, y_pred_group, labels=[0,1])

    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn + 1e-6)
    fnr = fn / (fn + tp + 1e-6)
    return {
        'Accuracy': acc,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr
    }

# Build fairness metrics
race_ids_test = [race_labels_new[i] for i in X_test_new.index]

fair_black = group_fairness_report(y_test_new, y_pred_new, race_ids_test, 'Black')
fair_white = group_fairness_report(y_test_new, y_pred_new, race_ids_test, 'White')

import pandas as pd
pd.DataFrame([fair_black, fair_white], index=['Black', 'White'])
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_embedding_clusters(embeddings, labels, title='Embedding Clusters'):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    reduced_df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    reduced_df['Race'] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Race', alpha=0.6, s=60)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Before mitigation
plot_embedding_clusters(X, race_labels, title='Before Mitigation')

# After mitigation
plot_embedding_clusters(X_new, race_labels_new, title='After Mitigation')
import numpy as np

def plot_accuracy_by_group(acc_black, acc_white):
    groups = ['Black', 'White']
    accuracies = [acc_black, acc_white]

    plt.figure(figsize=(6, 5))
    sns.barplot(x=groups, y=accuracies, palette='coolwarm')
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison by Race")
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.show()

# Example usage
plot_accuracy_by_group(fair_black['Accuracy'], fair_white['Accuracy'])
def plot_fairness_metrics(fair_black, fair_white):
    metrics = ['Accuracy', 'False Positive Rate', 'False Negative Rate']
    black_vals = [fair_black[m] for m in metrics]
    white_vals = [fair_white[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, black_vals, width, label='Black', color='black')
    ax.bar(x + width/2, white_vals, width, label='White', color='lightgrey')

    ax.set_ylabel('Rate')
    ax.set_title('Fairness Metrics by Race')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    for i in range(len(metrics)):
        ax.text(x[i] - width/2, black_vals[i] + 0.01, f"{black_vals[i]:.2f}", ha='center', color='white' if black_vals[i] > 0.5 else 'black')
        ax.text(x[i] + width/2, white_vals[i] + 0.01, f"{white_vals[i]:.2f}", ha='center')

    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.show()

plot_fairness_metrics(fair_black, fair_white)
