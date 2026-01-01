# 🐧 Session 1: The Morphology Classifier
**Objective:** Can we train a Machine Learning model to identify penguin species (Adelie, Chinstrap, Gentoo) based only on their physical measurements?

**The Data:**
- **Source:** Palmer Station, Antarctica LTER.
- **Variables:** Bill length, bill depth, flipper length, body mass, sex, island.

**The Mission:**
Use the Gemini AI agent to build a **Random Forest Classifier**. We want to:
1. Load the data from URL.
2. Clean missing values.
3. Split into Training (80%) and Testing (20%) sets.
4. Train the model.
5. Evaluate it (How often is it right?).
6. Visualize what the model "learned" (Feature Importance).

## Prompt
Type this into the Gemini chat pane or a comment in the code cell, then hit "Generate".
```
Write a Python script to load the penguins dataset from https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv. Drop rows with missing values. Convert the 'species' column to a numeric code if needed. Split the data into X (features) and y (target) and then into train/test sets (80/20). Train a Random Forest Classifier. Print the accuracy score and plot a confusion matrix using seaborn. Finally, plot a bar chart of the feature importances.
```

## Teaching Points

The "DropNA" Trap: Point out that `df.dropna()` removed rows. Ask: "Is this okay? What if we lost 50% of our data?" (In this dataset, we only lose a few rows, but it's a critical question).

Feature selection: Note that we only used the numeric columns (bill, flipper, etc.). Ask: "Would 'Island' be useful?" (Yes, because certain species only live on certain islands).

The Result: The accuracy should be very high (~97%).

Feature Importance: The plot usually shows Flipper Length or Bill Dimensions as the top predictors. This aligns with biological taxonomy.

## Code

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url)

# 2. Clean Data (Drop NaNs)
df = df.dropna()

# 3. Prepare X (Features) and y (Target)
# We drop 'species' (target) and 'island'/'sex' (categorical) for simplicity, 
# or we can encoding them. For a simple demo, let's stick to numeric measurements.
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Evaluate
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 8. Feature Importance
plt.figure(figsize=(8,4))
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', color='teal')
plt.title('What matters for Penguin ID?')
plt.show()
```
