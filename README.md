
# To tune or not to tune? A meta-leaning approach for recommending important hyperparameters

The following repository contains all metrails for repoducing the paper "To tune or not to tune? A meta-leaning approach for recommending important hyperparameters":

* the scripts for collecting performance data of 6 machine learning algorithms on 200 classification tasks from OpenML environment.
* the collected performance data of SVM, Decision Tree, Random Forest, AdaBoost, Gradient Boosting and Extra Trees Classifiers.
* Several notebooks that each performs one experiment and conducts the results.
* Based on PerformanceData, created new datasets that all are in output_csv folders.
* tools for:
  * Importing and modifying the collected data
  * Searching correlation between the dataset metafeatures and classifier performances.
  * Conducting statistical tests to compare performance of the classifiers over the tasks.
  * Computing the best value for each important hyperparameter.
  * Computing Wilcoxon test for verifing the result.

* script for extracting metafeatures of the datasets
* script for performing fANOVA on the performance data

## To start collecting data for a given classifier over all datasets

```python
from DataCollection.functions import *

path_to_datasets = 'Datasets/'
classification_per_algorithm(path=path_to_datasets, algorithm='DecisionTree')
```

## Conduct fANOVA on the data

```python
from fANOVA.fanova_functions import *
do_fanova(dataset_name='PerformanceData/AB_results_total.csv', algorithm='AdaBoost')
```

## Extract Metafeatures

```python
from tools.metafeatures import *
extract_for_all(path_to_datasets)
```

## Create the Database object to import the collected data in desired formats

```python
from Tools.database import Database
db = Database()
```

```python
per_dataset_acc = db.get_per_dataset_accuracies()
per_dataset_acc.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>AB</th>
      <th>ET</th>
      <th>RF</th>
      <th>DT</th>
      <th>GB</th>
      <th>SVM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AP_Breast_Omentum.csv</td>
      <td>0.981060</td>
      <td>0.976235</td>
      <td>0.976462</td>
      <td>0.973912</td>
      <td>0.983555</td>
      <td>0.914538</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AP_Breast_Prostate.csv</td>
      <td>0.995238</td>
      <td>0.995238</td>
      <td>0.995238</td>
      <td>0.995238</td>
      <td>0.995238</td>
      <td>0.961498</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AP_Endometrium_Lung.csv</td>
      <td>0.968363</td>
      <td>0.958392</td>
      <td>0.957018</td>
      <td>0.929240</td>
      <td>0.968363</td>
      <td>0.894591</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AP_Endometrium_Prostate.csv</td>
      <td>0.992857</td>
      <td>0.992857</td>
      <td>0.992857</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.984615</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AP_Endometrium_Uterus.csv</td>
      <td>0.854854</td>
      <td>0.837953</td>
      <td>0.859561</td>
      <td>0.827924</td>
      <td>0.860409</td>
      <td>0.758801</td>
    </tr>
  </tbody>
</table>
</div>


```python
metafeatures = db.get_metafeatures()
```
