# KNN (k-Nearest Neighbors)

An implementation and demonstration of the **k-Nearest Neighbors (KNN)** classification algorithm on the Iris dataset using `scikit-learn`.

---

## 📌 Algorithm Overview

**k-Nearest Neighbors (KNN)** is a non-parametric, instance-based **lazy learning algorithm**. Instead of learning an explicit model function during training, it memorizes the dataset and computes predictions dynamically during testing[cite: 3].

### Core Steps:
1. **Memorization:** Store the training feature vectors and corresponding class labels[cite: 3].
2. **Distance Calculation:** Compute distance metrics (e.g., Euclidean distance) between a query test point and all training instances[cite: 3].
3. **Neighbor Selection:** Identify the $k$ training points closest to the query point[cite: 3].
4. **Majority Voting:** Determine the target class based on the majority vote among the chosen $k$ neighbors[cite: 3].

> 💡 **Best Practice:** Select $k$ as an **odd number** to prevent tied votes during majority decision-making[cite: 3].

---

## 📊 Dataset Information

- **Dataset:** Iris Plants Dataset (`sklearn.datasets.load_iris`)[cite: 3]
- **Instances:** 150 total samples (50 per class)[cite: 3]
- **Features (4 numeric):**
  - Sepal Length (cm)[cite: 3]
  - Sepal Width (cm)[cite: 3]
  - Petal Length (cm)[cite: 3]
  - Petal Width (cm)[cite: 3]
- **Target Classes (3 categories):** `setosa` (0), `versicolor` (1), `virginica` (2)[cite: 3]

---

## 💻 Implementation & Workflow

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset
iris = load_iris()[cite: 3]
X = pd.DataFrame(iris.data, columns=iris.feature_names)[cite: 3]
y = iris.target[cite: 3]

# 2. Train-Test Split (80% Train, 20% Test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)[cite: 3]

# 3. Model Initialization (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)[cite: 3]

# 4. Training (Memorization)
knn.fit(x_train, y_train)[cite: 3]

# 5. Prediction
y_pred = knn.predict(x_test)[cite: 3]

# 6. Evaluation
accuracy = accuracy_score(y_test, y_pred)[cite: 3]
print(f"Model Accuracy: {accuracy * 100:.2f}%")[cite: 3]
