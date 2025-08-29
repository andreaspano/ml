
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # progress bar
from concurrent.futures import ProcessPoolExecutor, as_completed

def do_encode (df, categorical_cols):

    """
    Encode categorical variables

    Args:
        df (DataFrame): Input data frame.
        categorical_cols (array): array of cols to be encoded 

    Returns:
        DataFrame: DataFrame with encoded columns
    """

    # Get non-categorical columns
    non_categorical_cols = df.drop(columns=categorical_cols)

    # Fit the encoder
    encoder = OneHotEncoder(sparse_output=False)

    # Get new column names
    encoded = encoder.fit_transform(df[categorical_cols])
    
    # Convert encoded array to DataFrame
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    # Convert encoded array to DataFrame
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    # Combine the numeric and one-hot encoded columns
    df_final = pd.concat([non_categorical_cols, encoded_df], axis=1)

    #return
    return df_final


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
    scores = cross_val_score(clf, X, y, cv=cv)
    return np.mean(scores)


def evaluate_multiple_alpha(ccp_alpha, X, y , cv, n_workers):


    cv_scores = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_single_alpha, alpha, X, y , cv): alpha for alpha in ccp_alpha}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating alphas"):
            print(alpha)
            score = future.result()
            cv_scores.append(score)
    
    return cv_scores