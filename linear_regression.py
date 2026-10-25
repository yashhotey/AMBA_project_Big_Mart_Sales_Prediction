# linear regression evaluation on encoded datasets (noey and oey)
# here we are testing both versions of our encoded data to see which one performs better
# we compare predictions directly with the actual ground truth y_test labels

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

# here we load our actual sales target values
# y_train has the training sales figures that our model will learn from
# y_test has the unseen actual sales so we can calculate our real test error
y_train = pd.read_csv('Y_train.csv')['Item_Outlet_Sales']
y_test = pd.read_csv('Y_test.csv')['Item_Outlet_Sales']

# we need to drop these two columns because they are text identifiers
# linear regression works on straight line equations with numbers (beta * x)
# text strings like 'fda15' cannot be mathematically multiplied, so we exclude them
drop_cols = ['Item_Identifier', 'Outlet_Identifier']

def evaluate_linear_dataset(x_train_path, x_test_path, name):
    print(f'=== evaluating: {name} ===')
    
    # loading the preprocessed feature matrices
    x_train = pd.read_csv(x_train_path)
    x_test = pd.read_csv(x_test_path)

    # here we filter and pick only the numerical and encoded columns
    # this leaves us with clean inputs ready for linear algebra calculations
    features = [c for c in x_train.columns if c not in drop_cols]
    print(f'features used: {len(features)}')

    # here we initialize standard linear regression using ordinary least squares
    # this algorithm finds the best fit line that minimizes the sum of squared errors
    model = LinearRegression()
    
    # fitting the model so it learns the slope (coefficient) for each individual feature
    model.fit(x_train[features], y_train)

    # now we generate predictions on the unseen test set
    raw_preds = model.predict(x_test[features])
    
    # physical retail sales can never be negative in the real world
    # because a straight line can mathematically extend below zero, a few predictions become negative
    # so we use np.clip to set any negative values to zero as a domain correction step
    clipped_preds = np.clip(raw_preds, 0, None)

    # here we calculate standard regression evaluation metrics
    # rmse tells us our average prediction error in rupees (lower is better)
    # mae gives the average absolute error without heavily penalizing large outliers
    # r2 tells us what percentage of sales variance our linear equation explains
    rmse_raw = root_mean_squared_error(y_test, raw_preds)
    rmse_clipped = root_mean_squared_error(y_test, clipped_preds)
    mae = mean_absolute_error(y_test, clipped_preds)
    r2 = r2_score(y_test, clipped_preds)

    # printing out results so we can see the exact impact of non-negative clipping
    print(f'raw test rmse:     {rmse_raw:.2f}')
    print(f'clipped test rmse: {rmse_clipped:.2f}  (after non-negative clipping)')
    print(f'test mae:          {mae:.2f}')
    print(f'test r^2 score:    {r2:.4f}')
    print(f'negative predictions count: {(raw_preds < 0).sum()} out of {len(raw_preds)}')
    print()
    return model, features, clipped_preds

if __name__ == '__main__':
    # test 1: without outlet establishment years (noey)
    # here we test the dataset where outlet age was completely removed
    # we want to see if removing store age helps avoid any multicollinearity
    lr_noey, feat_noey, pred_noey = evaluate_linear_dataset(
        'NOEY/X_train_NOEY_final.csv',
        'NOEY/X_test_NOEY_final.csv',
        'NOEY (without outlet existed years)'
    )

    # test 2: with outlet establishment years (oey)
    # here we test the dataset where outlet existed years is kept as an active feature
    # this lets us verify whether older stores actually have a measurable sales advantage
    lr_oey, feat_oey, pred_oey = evaluate_linear_dataset(
        'OEY/X_train_OEY_final.csv',
        'OEY/X_test_OEY_final.csv',
        'OEY (with outlet existed years)'
    )

    # here we extract the learned regression weights (coefficients) from the best performing model (oey)
    # this makes our model fully explainable because we can see exactly which features drive sales the most
    coef_df = pd.DataFrame({'feature': feat_oey, 'coefficient': lr_oey.coef_})
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)

    # displaying the top 10 most influential features and their positive or negative impact
    print('top 10 strongest features in linear regression (oey):')
    for idx, row in coef_df.head(10).iterrows():
        print(f"  {row['feature']:30s} : {row['coefficient']:+10.2f}")
