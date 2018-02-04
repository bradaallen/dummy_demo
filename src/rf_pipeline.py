
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score


def data_preprocessing(dirty_iris):

    '''
    This function preprocesses the dirty data. We can also bake in all of the checks that were done
    throughout the analysis.
    This way, if the data changes, we'll be able to know quite quickly and easily.

    There are four steps of pre-processing:
    1. Cleaning column titiles
    2. Dropping outliers
    3. Converting errant values to centimeters
    4. Imputing mean for NaN values

    ** Parameters **
    -- dirty_iris: The data that is being loaded

    ** Returns **
    -- clean_iris: A dataframe that can be used in the ML workflow

    '''

    # clean up column titles
    dirty_iris.loc[dirty_iris['class'] == 'versicolor', 'class'] = 'Iris-versicolor'
    dirty_iris.loc[dirty_iris['class'] == 'Iris-setossa', 'class'] = 'Iris-setosa'

    # introduce an error check on the number of classes
    if dirty_iris['class'].nunique() == 3:
        pass
    else:
        return print("ERROR: Data contains more classes than expected.")

    # This line drops any 'Iris-setosa' rows with a separal width less than 2.5 cm
    dirty_iris = dirty_iris.loc[(dirty_iris['class'] != 'Iris-setosa') 
    							| (dirty_iris['sepal_width_cm'] >= 2.5)]

    # convert to centimeters
    dirty_iris.loc[(dirty_iris['class'] == 'Iris-versicolor') &
              (dirty_iris['sepal_length_cm'] < 1.0),
              'sepal_length_cm'] *= 100.0

    # mean imputation
    average_petal_width = dirty_iris.loc[dirty_iris['class'] == 'Iris-setosa', 
    															'petal_width_cm'].mean()

    dirty_iris.loc[(dirty_iris['class'] == 'Iris-setosa') &
              		   (dirty_iris['petal_width_cm'].isnull()),
              		   'petal_width_cm'] = average_petal_width
    
    # introduce an error check on Null handling
    if len(dirty_iris.loc[(dirty_iris['sepal_length_cm'].isnull()) |
              				(dirty_iris['sepal_width_cm'].isnull()) |
              				(dirty_iris['petal_length_cm'].isnull()) |
              				(dirty_iris['petal_width_cm'].isnull())].index) == 0:
        pass
    else:
        return print("ERROR: Null values still present after pre-processing.")

    clean_iris = dirty_iris

    return clean_iris

def data_checks(clean_iris):
    '''
    Alternatively (or in addition to), we can take our error handling and make it is own function
    that is run on the final dataset.
    '''
    # We know that we should only have three classes
    assert clean_iris['class'].nunique() == 3

    # We know that sepal lengths for 'Iris-versicolor' should never be below 2.5 cm
    assert clean_iris.loc[clean_iris['class'] == 'Iris-versicolor', 'sepal_length_cm'].min() >= 2.5

    # We know that our data set should have no missing measurements
    assert len(clean_iris.loc[(clean_iris['sepal_length_cm'].isnull()) |
                               (clean_iris['sepal_width_cm'].isnull()) |
                               (clean_iris['petal_length_cm'].isnull()) |
                               (clean_iris['petal_width_cm'].isnull())]) == 0

    return clean_iris


def rf_modeling_iris(clean_iris):
    '''
    This function takes our GridCV optimized RF function and then classifies it using the test data
    and a holdout

    Inputs:
    -- clean_iris: Pre-processed dataset of iris data

    Parameters:
    -- None

    Returns:
    -- random_forest_classifier: classifier object

    '''
    data_checks(clean_iris)

    all_inputs = clean_iris[['sepal_length_cm', 'sepal_width_cm',
                             'petal_length_cm', 'petal_width_cm']].values

    all_classes = clean_iris['class'].values

    # This is the classifier that came out of Grid Search
    random_forest_classifier = RandomForestClassifier(bootstrap=True, class_weight=None, 
    							criterion='gini', max_depth=None, max_features=3, max_leaf_nodes=None,
								min_samples_leaf=1, min_samples_split=2,
								min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=1,
								oob_score=False, random_state=None, verbose=0, warm_start=True)

    # All that's left to do now is plot the cross-validation scores
    rf_classifier_scores = cross_val_score(random_forest_classifier, all_inputs, all_classes, cv=10)
    sns.boxplot(rf_classifier_scores)
    sns.stripplot(rf_classifier_scores, jitter=True, color='white')

    # ...and show some of the predictions from the classifier
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = train_test_split(all_inputs, all_classes, train_size=0.75)

    random_forest_classifier.fit(training_inputs, training_classes)

    for input_features, prediction, actual in zip(testing_inputs[:10],
                                              random_forest_classifier.predict(testing_inputs[:10]),
                                              testing_classes[:10]):
        print('{}\t-->\t{}\t(Actual: {})'.format(input_features, prediction, actual))

    return random_forest_classifier

def main():
    '''
    docstrings
    '''

    iris_path = '../notebooks/data/iris-data.csv'
    iris_data = pd.read_csv(iris_path)
    clean_iris = data_preprocessing(iris_data)
    rf_classifier = rf_modeling_iris(clean_iris)

    return rf_classifier

if __name__ == "__main__":
    main()