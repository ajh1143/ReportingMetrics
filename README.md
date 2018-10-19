# ReportingMetrics
Shortcuts for automating reports about model performance

## Modules
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , classification_report

## Mean Absolute Error
def MeanAbsErr(y_test, y_pred):
    """
    Calculates and prints Mean Absolute Error
    :args: y_test
           y_pred
    :return none:
    """
    mean_err = metrics.mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error: {}'.format(round(mean_err), 3))

## Mean Squared Error
def MeanSqErr(y_test, y_pred):
    """
    Calculates and prints Mean Squared Error
    :args: y_test
           y_pred
    :return none:
    """
    SqErr = metrics.mean_squared_error(y_test, y_pred)
    print('Mean Squared Error: {}'.format(round(SqErr), 3))

## Model Score (Decision Tree Classification)
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

## Metric Report
def MetricReport(X, y, y_test, y_pred, dtc):
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
    ConfusionMatx(y_test, y_pred)
    MeanAbsErr(y_test, y_pred)
    MeanSqErr(y_test, y_pred)
    DTCScore(X, y, dtc)
    print("-" * 16)
