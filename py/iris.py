from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree

# Load example data
X, y = load_iris(return_X_y=True)

# Create and fit the tree
ct = DecisionTreeClassifier(criterion="gini", ccp_alpha=0.01253)


ct.fit(X, y)

# Predict
prd = ct.predict(X)


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
tree.plot_tree(ct, filled=True, feature_names=load_iris().feature_names, class_names=load_iris().target_names)
plt.show()

################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Step 1: Generate custom alpha sequence
ccp_alphas = np.linspace(0.00001, 0.02, 100)

# Step 2: Evaluate each alpha with cross-validation
cv_scores = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    scores = cross_val_score(clf, X, y, cv=5)
    cv_scores.append(np.mean(scores))

# Step 3: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, cv_scores, marker='o', drawstyle="steps-post")
plt.xlabel("ccp_alpha")
plt.ylabel("Cross-validated Accuracy")
plt.title("Cross-Validation for Custom ccp_alpha Values")
plt.grid(True)
plt.show()

# Step 4: Get the best alpha
best_alpha = ccp_alphas[np.argmax(cv_scores)]
best_score = max(cv_scores)
print(f"Best alpha: {best_alpha:.5f} with CV accuracy: {best_score:.4f}")


# Fit the best tree
clf_best = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha)
clf_best.fit(X, y)
# Plot using sklearn.tree.plot_tree
from sklearn.tree import plot_tree

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(
    clf_best,
    feature_names=X.columns,
    class_names=[str(cls) for cls in np.unique(y)],
    filled=True,
    rounded=True
)
plt.title(f"Decision Tree (ccp_alpha = {best_alpha:.5f})")
plt.show()
