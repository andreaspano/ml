
#Ref https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Build simple data set

df = pd.read_csv('../data/binary.csv')


# Fit the tree

## separate y & X
# Response 
y = df["y"]

# Predictors 
X = df.drop(columns=["y"])


trn_x, tst_x, trn_y, tst_y = train_test_split(X, y, random_state=0)

model = DecisionTreeClassifier(
    criterion="gini",
    min_samples_split = 2, 
    min_samples_leaf = 1,   
    min_impurity_decrease = 0,  
    max_depth=None,
    random_state=46, 
    ccp_alpha = 0)

# Fit the tree
model.fit(trn_x, trn_y)

# Plot Full Tree
#plt.figure(figsize=(12, 6))
plot_tree(
    model,
    filled=False,        
    feature_names = None,  
    class_names= None,    
    impurity=True,      # No impurity text
    proportion=False,    # No proportion text
    label='none',        # No node labels
    fontsize=0           # Effectively hides any residual text
)


# Compute Alpha working bottom up over the tree
path = model.cost_complexity_pruning_path(trn_x, trn_y)
ccp_alphas = path.ccp_alphas[:-1]
impurities = path.impurities[:-1]


# Plot Impurity vs Alpha
fig, ax = plt.subplots()
ax.plot(ccp_alphas, impurities, marker="o")
ax.set_xlabel("Effective Alpha")
ax.set_ylabel("Total Impurity of Leaves")
ax.set_title("Total Impurity vs Effective Alpha for Training Set")


# Check ccp effect
#ccp_alphas[ccp_alphas < 0 ] = 0
node_counts = []
depth = []
trn_scores = []
tst_scores = []

for ccp_alpha in ccp_alphas:
    tmp = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    tmp.fit(trn_x, trn_y)
    node_counts.append(tmp.tree_.node_count)
    depth.append(tmp.tree_.max_depth)
    trn_scores.append(tmp.score(trn_x, trn_y))
    tst_scores.append(tmp.score(tst_x, tst_y))


fig, ax = plt.subplots(2, 2)
ax[0,0].plot(ccp_alphas, node_counts, marker="o")
ax[0,0].set_xlabel("alpha")
ax[0,0].set_ylabel("number of nodes")
ax[0,0].set_title("Number of nodes vs alpha")

ax[0,1].plot(ccp_alphas, depth, marker="o" )
ax[0,1].set_xlabel("alpha")
ax[0,1].set_ylabel("depth of tree")
ax[0,1].set_title("Depth vs alpha")

ax[1,0].plot(ccp_alphas, impurities, marker="o")
ax[1,0].set_xlabel("alpha")
ax[1,0].set_ylabel("total impurity of leaves")
ax[1,0].set_title("Impurity vs alpha [trn]")

ax[1,1].plot(ccp_alphas, trn_scores, marker=".", label="train" )
ax[1,1].plot(ccp_alphas, tst_scores, marker=".", label="test")
ax[1,1].set_xlabel("Alpha")
ax[1,1].set_ylabel("Accuracy") #The .score() method of the classifier is fixed to return accuracy for classifiers.
ax[1,1].set_title("Accuracy vs alpha for trn and tst")
ax[1,1].legend()

fig.tight_layout()


# best test alphas
tst_alpha = ccp_alphas[np.argmax(tst_scores)]
print('Tst Alpha = ', tst_alpha)


# Better using Cross Val 
scores_avg = []
scores_std = []
cv = 5

for ccp_alpha in path.ccp_alphas:
    tmp = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(tmp, X, y, cv=cv, scoring="accuracy")  
    scores_avg.append(scores.mean())
    scores_std.append(scores.std())

#Pick the alpha with the best CV mean 
# Possibly use 1-SE rule)
best_idx   = np.argmax(scores_avg)
best_alpha = ccp_alphas[best_idx]
best_std = scores_std[best_idx]
best_cv    = scores_avg[best_idx] 


threshold = best_cv - best_std

# 3. Find the simplest tree (largest alpha) with accuracy >= threshold
candidate_indices = np.where(scores_avg >= threshold)[0]
one_se_idx = candidate_indices[-1]   # pick the largest alpha index

one_se_alpha = ccp_alphas[one_se_idx]
one_se_cv    = scores_avg[one_se_idx]

print(f"Best α (max CV): {best_alpha:.4f}, CV={best_cv:.3f}")
print(f"1-SE α: {one_se_alpha:.4f}, CV={one_se_cv:.3f}")


# Plot Cross Validation results
fig, ax = plt.subplots(1, 1)
plt.plot(path.ccp_alphas[:-1], scores_avg[:-1], marker="o" , color='blue')

# Plot with error bars
ax.errorbar(
    path.ccp_alphas[:-1],          # x-values
    scores_avg[:-1],               # y-values (means)
    yerr=scores_std[:-1],          # error bars (std dev)
    fmt="o-",                      # circle markers, connected by lines
    capsize=5,                     # small horizontal caps on error bars
    ecolor="green",                 # color of error bars
    elinewidth=1                   # thickness of error bars
)
ax.plot(best_alpha, best_cv, "ro", markersize=10, label="Best α")  
ax.annotate(
    f"Best α={best_alpha:.3f}\nAcc={best_cv:.3f}",
    (best_alpha, best_cv),
    textcoords="offset points",
    xytext=(10,10),
    ha="left",
    color="red"
)
ax.plot(one_se_alpha, one_se_cv, "ro", markersize=10, label="Best one se α")  
ax.annotate(
    f"Best α={one_se_alpha:.3f}\nAcc={one_se_cv:.3f}",
    (one_se_alpha, one_se_cv),
    textcoords="offset points",
    xytext=(10,10),
    ha="left",
    color="red"
)

ax.set_xlabel("Alpha")
ax.set_ylabel("Cross Validation Accuracy")
ax.set_title("CV Accuracy vs Alpha")
plt.show()



# Refit with best alpha
best_model = DecisionTreeClassifier(
    criterion="gini",
    random_state=46, 
    ccp_alpha = best_alpha) 
# Fit the tree
best_model.fit(trn_x, trn_y)


# Refit with one_se_alpha
one_se_model = DecisionTreeClassifier(
    criterion="gini",
    random_state=46, 
    ccp_alpha = one_se_alpha) #best_alpha
# Fit the tree
one_se_model.fit(trn_x, trn_y)



# Plot Full Tree
#plt.figure(figsize=(12, 6))
plot_tree(
    best_model,
    filled=False,        
    feature_names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],  
    class_names=['0','1'],    
    impurity=True,      # No impurity text
    proportion=False,    # No proportion text
    label='none',        # No node labels
    fontsize=10           # Effectively hides any residual text
)



# Plot best_tree & one_se tree
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot first tree
plot_tree(
    best_model,
    ax=axes[0],
    filled=False,        
    feature_names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],  
    class_names=['0','1'],    
    impurity=True,      # No impurity text
    proportion=False,    # No proportion text
    label='none',        # No node labels
    fontsize=10           # Effectively hides any residual text
)
axes[0].set_title("Best Model")

# Plot second tree
plot_tree(
    one_se_model,
    ax=axes[1],
    filled=False,        
    feature_names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],  
    class_names=['0','1'],    
    impurity=True,      # No impurity text
    proportion=False,    # No proportion text
    label='none',        # No node labels
    fontsize=10           # Effectively hides any residual text
)
axes[1].set_title("One se Model")

plt.show()
