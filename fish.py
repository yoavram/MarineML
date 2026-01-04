import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load (Update URL to your GitHub Raw Link after uploading)
url = "Fish.csv"
df = pd.read_csv(url)

# 2. Cleaning (Real Data Check)
# Check for "Ghost Fish" (Weight = 0 or negative)
print(f"Original Count: {len(df)}")
df = df[df['Weight'] > 0] 
print(f"Cleaned Count: {len(df)}")

# 3. Ecological Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(x='Species', y='Weight', data=df, palette='Spectral')
plt.title("Biomass Distribution by Species (Trophic Level Proxy)")
plt.yscale('log') # Log scale helps visualize small Smelt vs huge Pike
plt.ylabel("Estimated Biomass (g)")
plt.show()

# 4. Train Model
X = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 5. Evaluate
print("Accuracy:", rf.score(X_test, y_test))

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, rf.predict(X_test)), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title("Classification Errors")
plt.show()