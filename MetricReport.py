import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , classification_report

def MeanAbsErr(y_test, y_pred):
    """
    Calculates and prints Mean Absolute Error
    :args: y_test
           y_pred
    :return none:
    """
    mean_err = metrics.mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error: {}'.format(round(mean_err), 3))

def MeanSqErr(y_test, y_pred):
    """
    Calculates and prints Mean Squared Error
    :args: y_test
           y_pred
    :return none:
    """
    SqErr = metrics.mean_squared_error(y_test, y_pred)
    print('Mean Squared Error: {}'.format(round(SqErr), 3))

def DTCScore(X, y, dtc):
    """
    Calculates and prints model score
    :args: X - Features
           y - Target
           dtc - Decision Tree Classifier Instance
    :return none:
    """
    score = dtc.score(X, y, sample_weight=None)
    print('Score: {}'.format(round(score)))
    
def feature_finder(df, model):
    """
    Calculates and prints feature importance
    :args: df - dataframe of dataset
           model - fitted model
    :return none:
    """
    features = dict(zip(df.columns, model.feature_importances_))
    print(features)

def MetricReport(df, X, y, y_test, y_pred, dtc, model):
    """
    Compiles a report of performance metrics
    :args: X - Features
           y - Target
           y_test
           y_pred
           dtc - Decision Tree Classifier Instance
    :return none:
    """
    print("Metric Summaries")
    print("-"*16)
    feature_finder(df, model)
    ConfusionMatx(y_test, y_pred)
    MeanAbsErr(y_test, y_pred)
    MeanSqErr(y_test, y_pred)
    DTCScore(X, y, dtc)
    print("-" * 16)
