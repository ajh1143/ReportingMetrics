# Model Reporting Metrics
Shortcuts for automating reports about model performance


## Modules
```Python
import graphviz
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , classification_report
```


## Mean Absolute Error
```Python3
def MeanAbsErr(y_test, y_pred):
    """
    Calculates and prints Mean Absolute Error
    :args: y_test
           y_pred
    :return none:
    """
    mean_err = metrics.mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error: {}'.format(round(mean_err), 3))
```


## Mean Squared Error
```Python3
def MeanSqErr(y_test, y_pred):
    """
    Calculates and prints Mean Squared Error
    :args: y_test
           y_pred
    :return none:
    """
    SqErr = metrics.mean_squared_error(y_test, y_pred)
    print('Mean Squared Error: {}'.format(round(SqErr), 3))
```


## Model Score (Decision Tree Classification)
```Python3
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
```


## Feature Finder
```Python3
def feature_finder(df, model):
    """
    Calculates and prints feature importance
    :args: df - dataframe of dataset
           model - fitted model
    :return none:
    """
    features = dict(zip(df.columns, model.feature_importances_))
    print(features)
```


## Tree Graph Visualization
```Python3
def tree_viz(dtc, df, col_names, class_names, title):
    """
    Generates a tree graph visualization
    :args: 
           dtc - decision tree instance
           df - dataframe of dataset
           col_names - list of column names
           class_names - list of classification names
           title - name of graph dataset
    :return none:
    """
    class_n = class_names
    dot = tree.export_graphviz(dtc, out_file=None, feature_names=col_names, class_names=class_n, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot)
    graph.format = 'png'
    graph.render(title, view=True)
```


## Report Generator
```Python3
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
```
