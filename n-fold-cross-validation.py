import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# read and shuffle data, create test split
dataset = pd.read_csv("data_banknote_authentication.csv")
shuffled_df = dataset.sample(frac=1, random_state=1).reset_index(drop=True)
train, validate = train_test_split(shuffled_df, test_size=.2)

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
weights_list = []

# create lists for each of the attributes to calc the average of them later
a1_weights = []
a2_weights = []
a3_weights = []
a4_weights = []
intercepts = []
classes = []

# iterate through the folds and use each fold once as the test set 
for idx in range(len(folds)):
    print(f"    FOLD NUMBER {idx+1}")
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

    a1_weights.append(log_reg.coef_[0][0])
    a2_weights.append(log_reg.coef_[0][1])
    a3_weights.append(log_reg.coef_[0][2])
    a4_weights.append(log_reg.coef_[0][3])
    intercepts.append(log_reg.intercept_)
    classes = log_reg.classes_

    print(f"""
    
    accuracy score is: {accuracy}
    
    f1 score is: {f1}
    
    """)


# calculate the average of each weight
a1_average = sum(a1_weights)/len(a1_weights)
a2_average = sum(a2_weights)/len(a2_weights)
a3_average = sum(a3_weights)/len(a3_weights)
a4_average = sum(a4_weights)/len(a4_weights)
intercept_average = sum(intercepts)/len(intercepts)


# create average weights array and initialize a model using them
average_weights = np.array([[a1_average,a2_average,a3_average,a4_average]])
log_reg_final = LogisticRegression(C=1)
log_reg_final.coef_ = average_weights
log_reg_final.intercept_ = intercept_average
log_reg_final.classes_ = classes

validate = pd.DataFrame(data=validate)
y_validate = validate["class"].to_numpy()
x_validate = validate.drop(["class"], axis=1).to_numpy()

y_pred = log_reg_final.predict(x_validate)
accuracy = round(accuracy_score(y_validate,y_pred)*100, 2)
f1 = round(f1_score(y_validate,y_pred)*100, 2)

print(f"""
FINAL MODEL TESTED ON VALIDATION SET:

    accuracy score is: {accuracy}
    
    f1 score is: {f1}
    
""")
