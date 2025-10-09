#Ref https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm  # progress bar
from concurrent.futures import ProcessPoolExecutor, as_completed


data = load_breast_cancer(as_frame=True)
print(data.keys())
data.DESCR

X = data.data
y = data.target





X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(random_state=0, criterion = 'gini', ccp_alpha=0.0)


# Fit the tree
clf.fit(X, y)

# Plot Fully growth Tree
plt.figure(figsize=(12, 6))
plot_tree(
    clf,
    filled=False,        # No coloring
    feature_names=None,  # No feature names
    class_names=None,    # No class names
    impurity=False,      # No impurity text
    proportion=False,    # No proportion text
    label='none',        # No node labels
    fontsize=0           # Effectively hides any residual text
)
#plt.show()

# Number of terminal leaves
n_leaves = clf.tree_.n_leaves
print("Number of terminal leaves:", n_leaves)


# Total numer of nodes
total_nodes = clf.tree_.node_count
print("Total number of nodes:", total_nodes)

depth = clf.get_depth()+1 # (root has depth 0)
print("Tree depth:", depth)

'''
cost_complexity_pruning_path internally grows a full decision tree 
before returning the sequence of ccp_alpha values and impurities 
so it does fit the tree under the hood, even if you never call clf.fit() explicitly.

The path call is just for analysis of possible pruning levels, 
not for creating a ready-to-use classifier

Step 1: It grows the maximal tree 
(no pre-pruning, until leaves are pure or can’t be split further).

Step 2: From that single tree, it computes the sequence of possible 
ccp_alpha values and the corresponding total impurities.

Step 3: It doesn’t build a new tree for each alpha — 
it simply simulates pruning by collapsing internal nodes into leaves step by step, 
recalculating impurity changes as it goes.
'''

path = clf.cost_complexity_pruning_path(X_train, y_train)


# collect alphas and impurities
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# Plot the total impurity vs alpha values
fig, ax = plt.subplots()
ax.plot(impurities[:-1], ccp_alphas[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("total impurity of leaves")
ax.set_ylabel("effective alpha")
ax.set_title("Effective alpha vs Total Impurity for training set")
ax.invert_xaxis()


#clfs = []
node_counts = []
depth = []
train_scores = []
test_scores = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha, criterion="gini")
    clf.fit(X_train, y_train)
    #clfs.append(clf)
    node_count = clf.tree_.node_count
    max_depth = clf.tree_.max_depth
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)


    node_counts.append(node_count)
    depth.append(max_depth)
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    


node_counts = node_counts[:-1]
depth = depth[:-1]
train_scores = train_scores[:-1]
test_scores = test_scores[:-1]
ccp_alphas = ccp_alphas[:-1]




#clfs = clfs[:-1]

#node_counts = [clf.tree_.node_count for clf in clfs]
#depth = [clf.tree_.max_depth for clf in clfs]




fig, ax = plt.subplots(2, 1)
ax[0].plot( node_counts, ccp_alphas, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("number of nodes")
ax[0].set_ylabel("alpha")
ax[0].set_title("Alpha vs Number of nodes")


ax[1].plot(depth, ccp_alphas, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("depth of tree")
ax[1].set_ylabel("alpha")
ax[1].set_title("Alpha vs depth ")


fig.tight_layout()




#train_scores = [clf.score(X_train, y_train) for clf in clfs]
#test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
ax.invert_xaxis()
plt.show()

# Pick best aplha
best_alpha = ccp_alphas[np.argmax(test_scores)]
print("Best alpha:", best_alpha)


# Fit model with optimum alpha
clf = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha, criterion="gini")
clf.fit(X_train, y_train)

# Plot best model
plt.figure(figsize=(12, 6))
plot_tree(
    clf,
    filled=True,        # No coloring
    feature_names=data.feature_names,  # No feature names
    class_names=None,    # No class names
    impurity=False,      # No impurity text
    proportion=False,    # No proportion text
    label='none',        # No node labels
    fontsize=13           # Effectively hides any residual text
)


##################################################
#Cross Validation


def evaluate_single_alpha(ccp_alpha, X, y , cv):

    """
    Fits a DT y~X given the value of ccp_alpha
    Compute cross_val_score for cv folds 
    Return the mean of cross_val_score across cross_val_score

    Args:
        ccp_alpha (float): ccp alpha 
        X (DataFrrame): Refressor matrix
        y (array): response vector
        cv (int): number of cross validation sets 
    Returns:
        float: Mean of cross_val_score across cross_val_score
    """
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    return np.mean(scores)

evaluate_single_alpha(ccp_alpha = best_alpha, X = X_train,  y = y_train , cv = 5)



def evaluate_multiple_alpha(ccp_alpha, X, y , cv, n_workers):
    cv_scores = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_single_alpha, alpha, X, y , cv): alpha for alpha in ccp_alpha}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating alphas"):
            score = future.result()
            cv_scores.append(score)
    
    return cv_scores


accuracy = evaluate_multiple_alpha(ccp_alpha = ccp_alphas, X = X_train,  y = y_train , cv = 8 , n_workers = 8)


# Plot alpha vs cv accuracy
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("CV Accuracy")
ax.set_title("CV Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, accuracy, marker="o", label="cv train", drawstyle="steps-post")
#ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
ax.invert_xaxis()
plt.show()


best_alpha = ccp_alphas[np.argmax(accuracy)]
print("Best alpha:", best_alpha)

# Fit model with optimum alpha
clf = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha, criterion="gini")
clf.fit(X_train, y_train)

# Plot best model
plt.figure(figsize=(12, 6))
plot_tree(
    clf,
    filled=True,        # No coloring
    feature_names=data.feature_names,  # No feature names
    class_names=None,    # No class names
    impurity=False,      # No impurity text
    proportion=False,    # No proportion text
    label='none',        # No node labels
    fontsize=13           # Effectively hides any residual text
)

# Grid search cv

from sklearn.model_selection import GridSearchCV

# reuse ccp_alphas from the pruning path
grid = GridSearchCV(
    DecisionTreeClassifier(), #random_state=0
    param_grid={"ccp_alpha": ccp_alphas},
    scoring="accuracy",
    cv=5
)
grid.fit(X, y)
print("Best alpha:", grid.best_params_["ccp_alpha"])
print("CV score:", grid.best_score_)
best_clf = grid.best_estimator_

# test 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

# 6. Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="Blues", values_format='d')
