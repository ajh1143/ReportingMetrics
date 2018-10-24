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
Calculates the *MAE*, prints in the form: `Mean Absolute Error: 0.123`    
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
Calculates the *MSE*, prints in the form `Mean Squared Error: 0.123`    
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
Calculates *Accuracy Score*, prints in the form `Score: 0.1`    
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
Calculates the importance of features in the model, in the form `{'Feature_1': 0.1, 'Feature_2': 0.2, 'Feature_3': 0.3}`    
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

## Plot Feature Weights
```Python3
def plot_features(feature_dict):
    '''
    :param feature_dict: Dictionary of feature weights in k,v pairs
    :retur: None, Displays bar plot of features and model weights
    '''
    feature_dict = dict((k, v) for k, v in feature_dict.items() if v >= 0.01)
    names = list(feature_dict.keys())
    values = list(feature_dict.values())
    values = values
    plt.bar(names, values)
    plt.xlabel('Categories')
    plt.ylabel('Percentage\n(%)')
    plt.title('Feature Weight')
    plt.show()
```

## Tree Graph Visualization    
Creates a visualization of the decision tree, in the form of nodes and branches with attributes tested, classes etc.    

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
Creates a report of metrics, prints line-by-line results

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
