'''
Diabets data
Diabetes_012 =>	Target: 0 = no diabetes, 1 = prediabetes, 2 = diabetes
HighBP =>	High blood pressure (0/1)
HighChol =>	High cholesterol (0/1)
CholCheck =>	Had cholesterol check (0/1)
BMI	=> Body Mass Index
Smoker, Stroke, HeartDiseaseorAttack => Health indicators
PhysActivity, Fruits, Veggies, HvyAlcoholConsump	=> Health behaviors
AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk	=> Healthcare access and health status
Sex, Age, Education, Income	=> Socio-demographic variables
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
#from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
#from sklearn.model_selection import cross_val_score
#from concurrent.futures import ProcessPoolExecutor
#from tqdm import tqdm  # progress bar
#from concurrent.futures import ProcessPoolExecutor, as_completed
from defs import * 

## parameters
min_alpha = 0.0005
max_alpha = 0.01
n_alpha = 5
n_workers = 12
cv = 5
## Function 



# read data
df = pd.read_csv('../data/diabetes.csv', sep=",")

# define categorical columns
categorical_cols= ['HighBP','HighChol','CholCheck','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex','Education']

# encode cataegorical variables
df = do_encode (df, categorical_cols)


# Response 
y = df["Diabetes_012"]

# Predictors 
X = df.drop(columns=["Diabetes_012"])

# Generate custom alpha sequence
ccp_alpha = np.linspace(min_alpha, max_alpha, n_alpha)

# Run in parallel
cv_scores =  evaluate_multiple_alpha(ccp_alpha, X, y , cv, n_workers)


# Step 3: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(ccp_alpha, cv_scores, marker='o', drawstyle="steps-post")
plt.xlabel("ccp_alpha")
plt.ylabel("Cross-validated Accuracy")
plt.title("Cross-Validation for Custom ccp_alpha Values")
plt.grid(True)
plt.show()

# Step 4: Get the best alpha
best_alpha = ccp_alpha[np.argmax(cv_scores)]
best_score = max(cv_scores)
print(f"Best alpha: {best_alpha:.5f} with CV accuracy: {best_score:.4f}")
