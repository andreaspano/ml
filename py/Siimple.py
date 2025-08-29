
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.DataFrame({
    "y" : [1,1,1,1,1,0,0,0,0,0], 
    "x1" : [0,0,0,0,1,1,1,1,1,1],
    "x2" : [0,1,0,1,0,1,0,1,0,1]
 })


# Response 
y = df["y"]

# Predictors 
X = df.drop(columns=["y"])


# Model
model = DecisionTreeClassifier(
    criterion="gini",
    min_samples_split = 2, 
    min_samples_leaf = 1,   
     min_impurity_decrease = 0,  
    max_depth=None,
    random_state=46, 
    ccp_alpha = 0)



# Fit
model.fit(X, y)


# Plot basi tree
#fig, ax = plt.subplots(figsize=(7,5), facecolor = "white")

plot_tree(
    model,
    filled=False,        # No coloring
    feature_names=['x1', 'x2'],  # No feature names
    class_names=None,    # No class names
    impurity=True,      # No impurity text
    proportion=False,    # No proportion text
    label="all",        # No node labels
    fontsize=10
    
)
#plt.close(fig) 

#display(fig)


# Pruning 
path = model.cost_complexity_pruning_path(X, y)

ccp_alphas =  path.ccp_alphas 
impurities = path.impurities



# Plot alphas
fig, ax = plt.subplots()
ax.plot(impurities, ccp_alphas, marker="o", drawstyle="steps-post")
ax.set_ylabel("effective alpha")
ax.set_xlabel("total impurity of leaves")

# Etichette sopra ogni punto
for xx, yy in zip(impurities, ccp_alphas):
    ax.annotate(f"{xx:.3f}, {yy:.3f}",  # impurità con 3 decimali, alpha in notazione scientifica
                xy=(xx, yy),
                xytext=(0, 6),        # spostamento in pixel verso l'alto
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10)



###############################################################
# test trees with different alphas


fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid
axes = axes.ravel()  # flatten to 1D for easy looping

for i, alpha in enumerate(ccp_alphas):
    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=46,
        ccp_alpha=alpha
    )
    model.fit(X, y)

    plot_tree(
        model,
        filled=True,
        feature_names=['x1','x2'],
        class_names=['0','1'],
        impurity=True,
        proportion=False,
        label="all",
        fontsize=8,
        ax=axes[i]
    )
    axes[i].set_title(f"ccp_alpha = {alpha:.4f}")

axes[3].plot(impurities, ccp_alphas, marker="o", drawstyle="steps-post")
axes[3].set_ylabel("effective alpha")
axes[3].set_xlabel("total impurity of leaves")
axes[3].set_title("Cost-Complexity Pruning Path")

# Annotate points
for xx, yy in zip(impurities, ccp_alphas):
    axes[3].annotate(f"{xx:.3f}, {yy:.3f}",
                     xy=(xx, yy),
                     xytext=(0, 6),
                     textcoords="offset points",
                     ha="center", va="bottom",
                     fontsize=8)
axes[3].axis("off")

#plt.tight_layout()
plt.show()

####################

## cross val
from sklearn.model_selection import train_test_split, cross_val_score

model
scores = cross_val_score(model, X, y, cv=2, scoring='accuracy')
   
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')