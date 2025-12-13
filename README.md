# Wine Quality Analysis (White Wine)
This Machine learning project analyses the quality of white wine based on physicochemical characteristics, with quality scores ranging from 3 to 9. It applies multiple supervised classification models, with hyperparameter tuning performed for selected models.

## Models Implemented
The Supervised learning models for classification used in the project are as follows:
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- Random Forest Classifier
  

Hyperparameter tuning was used for SVM and Random Forest. The training was conducted on 70% of the dataset while 30% of it was used in testing.

## Dataset 
The UCI Wine Quality (white wine) dataset was used for this project: https://archive.ics.uci.edu/dataset/186/wine+quality  
The dataset has 4898 white vinho verde wine samples with 11 physicochemical properties. The target class ranges for a quality index of 3 to 9 with quality 5 and 6 dominating the dataset and 3 and 9 being least occurred.   
Among all features, alcohol content shows the strongest correlation with wine quality.

## Workflow
1. Load the UCI winequality-white.csv
2. Perform Exploratory Data Analysis on the dataset.
3. Split the data into training and testing using train_test_split.
4. Scale the feature data using StandardScaler (fit on training data).
5. Train LogisticRegression, DecisionTree, SVM, GradientBoosting and RandomForest.
6. Tune for SVM and RandomForest and look for best hyperparameters that improve the accuracy.
7. Predict on test data and measure accuracy scores.
8. Compare model accuracies.

## Results
| Model                    | Accuracy |
|-------------------------|----------|
| Logistic Regression     | 0.5530   |
| Decision Tree           | 0.5829   |
| SVM (tuned)             | 0.5836   |
| Gradient Boosting       | 0.5775   |
| Random Forest (baseline)| 0.6775    |
| Random Forest (final)   | 0.6680    |  

Note: Hyperparameter tuning did not improve Random Forest performance beyond the baseline model.


## Libraries Used  
Python 3
numpy  
pandas  
matplotlib 
seaborn  
scikit-learn

## How to Run
### Option 1- Google Colab
Open the Notebook:
```
wine_quality_analysis.ipynb
```
Download the winequality-white.csv dataset and upload it to colab.
```
uploaded = files.upload()
```
Install dependencies inside a cell
```
!pip install -r requirements.txt
```
Run the notebook cell by cell.
### Option 2- Local Jupyter Notebook
Install dependencies:
```
pip install -r requirements.txt
```
Launch Jupyter:
```
jupyter notebook
```
Open notebook and run all cells.

## File Structure
wine-quality-analysis/  
├── wine_quality_analysis.ipynb  
├── requirements.txt  
└── README.md












