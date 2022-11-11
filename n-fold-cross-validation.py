import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# read and shuffle data, create test split
dataset = pd.read_csv("data_banknote_authentication.csv")
shuffled_df = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
train, test = train_test_split(shuffled_df, test_size=.2)

# split the data frame into k folds
def split_dataframe_by_position(df: pd.DataFrame, splits: int) -> list[pd.DataFrame]:
    dataframes = []
    index_to_split = len(df) // splits
    start = 0
    end = index_to_split
    for split in range(splits):
        temporary_df = df.iloc[start:end, :]
        dataframes.append(temporary_df)
        start += index_to_split
        end += index_to_split
    return dataframes

folds = split_dataframe_by_position(train, 10)
print(len(folds))
weights_list = []

# iterate through the folds and use each fold once as the test set 
for idx in range(len(folds)):
    curr_test = folds[idx]
    train_p1 = folds[:idx]
    train_p2 = folds[idx+1:]
    train_list = train_p1 + train_p2
    train = pd.concat(train_list)

    # create x_train, y_train and x_test, y_test
    y_train = train["class"].to_numpy()
    x_train = train.drop(["class"], axis=1).to_numpy()
    y_test = curr_test["class"].to_numpy()
    x_test = curr_test.drop(["class"], axis=1).to_numpy()

    
    # train log reg model and predict on test data
    log_reg = LogisticRegression(C=1)
    log_reg.fit(x_train,y_train)

    y_pred = log_reg.predict(x_test)

    accuracy = round(accuracy_score(y_test,y_pred)*100, 2)
    f1 = round(f1_score(y_test,y_pred)*100, 2)
    weights_list.append(log_reg.coef_)

    print(f"""
    
    accuracy score is: {accuracy}
    
    f1 score is: {f1}
    
    """)

print(weights_list)
# TODO: Average the weights in the weights list
def take_average(wl: list[np.ndarray]) -> np.ndarray:
    average = np.zeros(len(wl[0]))
    for array in wl:
        print(array)
        
    
    return wl[0]
