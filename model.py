import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import joblib

train_data_xgb = pd.read_excel('train_with_features_xgb.xlsx')
test_data_xgb = pd.read_excel('test_with_features_xgb.xlsx')
train_data_forest = pd.read_excel('train_with_features_forest.xlsx')
test_data_forest = pd.read_excel('test_with_features_forest.xlsx')


y_train = pd.read_excel('y_train.xlsx')


y_test_xgb_max = pd.DataFrame()
y_test_xgb_mean = pd.DataFrame()
y_test_forest_max = pd.DataFrame()
y_test_forest_mean = pd.DataFrame()

y_test_xgb_max['Variant name'] = test_data_xgb['Variant number']
y_test_forest_max['Variant name'] = test_data_xgb['Variant number']
y_test_xgb_mean['Variant name'] = test_data_forest['Variant number']
y_test_forest_mean['Variant name'] = test_data_forest['Variant number']


test_data_xgb.drop(columns = ['Variant sequence','Variant number'],inplace =True)
test_data_forest.drop(columns = ['Variant sequence','Variant number'],inplace =True)


train_data_xgb = train_data_xgb.iloc[:,1:]
train_data_forest = train_data_forest.iloc[:,1:]

test_data_xgb = test_data_xgb.iloc[:,1:]
test_data_forest = test_data_forest.iloc[:,1:]

#uploading the features table into dataframes
columns = train_data_xgb.columns
for col in columns:
    test_data_xgb[col] = test_data_xgb[col].apply(lambda x:float(x))
    train_data_xgb[col] = train_data_xgb[col].apply(lambda x:float(x))
columns = train_data_forest.columns
for col in columns:
    test_data_forest[col] = test_data_forest[col].apply(lambda x:float(x))
    train_data_forest[col] = train_data_forest[col].apply(lambda x:float(x))
print(test_data_xgb.columns)

columns_y = y_train.columns
for col in columns_y:
    y_train[col] = y_train[col].apply(lambda x: float(x))

#function to caculate spearmen score
def spearman_scorer(y_true, y_pred):
    # Convert inputs to DataFrame if they are not already
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)

    # Compute Spearman correlation for each column
    spearman_scores = [spearmanr(y_true.iloc[:, 0], y_pred.iloc[:, 0]).correlation,
                       spearmanr(y_true.iloc[:, 1], y_pred.iloc[:, 1]).correlation]

    # Return the mean of Spearman correlations
    return np.mean(spearman_scores)

custom_scorer = make_scorer(spearman_scorer, greater_is_better=True)





#removing index column
y_train = y_train.iloc[:,1:]



#finding hyper parametes for the xgb boost
#param_grid = {
    #'n_estimators': [50, 100, 200],
    #'max_depth': [3, 4, 5, 6],
    #'learning_rate': [0.01, 0.1, 0.2],
    #'subsample': [0.6, 0.8, 1.0],
    #'colsample_bytree': [0.6, 0.8, 1.0],
    #'reg_alpha': [0, 0.1, 0.5],
    #'reg_lambda': [1, 1.5, 2]}



# Create the scorer
#spearman_scorer = make_scorer(spearman_scorer)

#grid_search = GridSearchCV(
    #estimator=xgbmodel,
    #param_grid=param_grid,
    #scoring=custom_scorer,
    #cv=5,  # 5-fold cross-validation
    #n_jobs=-1,  # Use all available cores
    #verbose=2)

#finding the best hyperparameters with gridsearch and  crossvalidation
#grid_search.fit(train_data_xgb, y_train)
#print("Best parameters:", grid_search.best_params_)
#print("Best Spearman correlation:", grid_search.best_score_)
#the model with the optimal parameters
best_model_xgb  = xgb.XGBRegressor(objective='reg:squarederror',colsample_bytree = 0.8, learning_rate =  0.2, max_depth =  4, n_estimators = 200, reg_alpha =  0, reg_lambda =2,subsample =0.8)
best_model_xgb.fit(train_data_forest,y_train)
y_pred_xgb = best_model_xgb.predict(test_data_xgb)


#Best Spearman correlation: 0.3398849981083468
#Best parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.2, 'max_depth': 4, 'n_estimators': 200, 'reg_alpha': 0, 'reg_lambda': 2, 'subsample': 0.8}

loaded_model_xgb = joblib.load("best_model_xgb.joblib")
y_pred_xgb = loaded_model_xgb.predict(test_data_xgb)

y_pred_xgb = pd.DataFrame(y_pred_xgb)
y_test_xgb_max['max diff'] = (pd.DataFrame(y_pred_xgb)).iloc[:,0]
y_test_xgb_max.sort_values(by='max diff', inplace = True)
y_test_xgb_mean['mean diff'] = (pd.DataFrame(y_pred_xgb)).iloc[:,1]
y_test_xgb_mean.sort_values(by='mean diff', inplace = True)
#predict random forest

model_forest = RandomForestRegressor(n_estimators = 100,random_state=42)
model_forest.fit(train_data_forest, y_train)

# Make predictions random forest
y_predforest = model_forest.predict(test_data_forest)
spearman_scorer = make_scorer(spearman_scorer)

# Define the parameter grid for RandomForestRegressor
#param_grid = {
    #'n_estimators': [50, 100, 200],
    #'max_depth': [3, 4, 5, 6],
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4],
    #'bootstrap': [True, False]}


# Create the RandomForestRegressor model
#model_forest2 = RandomForestRegressor()
# Create the GridSearchCV object
#grid_search = GridSearchCV(
    #estimator=model_forest2,
    #param_grid=param_grid,
    #scoring= custom_scorer,
    #cv=5,  # 5-fold cross-validation
    #n_jobs=-1,  # Use all available cores
    #verbose=2
#)

#find hte best parameters for our model
#grid_search.fit(train_data_forest,y_train)
#print("Best parameters:", grid_search.best_params_)
#print("Best Spearman correlation:", grid_search.best_score_)

# Use the best model for predictions
best_model_forest = RandomForestRegressor(bootstrap = True, max_depth =  6, min_samples_leaf =1, min_samples_split = 2, n_estimators =  100)
best_model_forest.fit(train_data_forest,y_train)
y_pred_forest = best_model_forest.predict(test_data_forest)

# Save the model
joblib.dump(best_model_forest, "best_model_forest.joblib")
loaded_model_forest = joblib.load("best_model_forest.joblib")

y_pred_forest = loaded_model_forest.predict(test_data_forest)
y_pred_forest = pd.DataFrame(y_pred_forest)




y_test_forest_max['max diff'] = (pd.DataFrame(y_pred_forest)).iloc[:,0]
y_test_forest_max.sort_values(by='max diff', inplace = True)

y_test_forest_mean['mean diff'] =( pd.DataFrame(y_pred_forest)).iloc[:,1]
y_test_forest_mean.sort_values(by='mean diff', inplace = True)

#Best parameters: {'bootstrap': True, 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
#Best Spearman correlation: 0.2250217643364612


with pd.ExcelWriter('final_predictions2.xlsx') as writer:
    y_test_xgb_max.to_excel(writer, sheet_name='max diff xgb', index=False)
    y_test_xgb_mean.to_excel(writer, sheet_name='mean diff xgb ', index=False)
    y_test_forest_max.to_excel(writer, sheet_name='max random forest', index=False)
    y_test_forest_mean.to_excel(writer, sheet_name='diff mean random forest', index=False)






