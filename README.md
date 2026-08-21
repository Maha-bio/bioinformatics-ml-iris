Iris Machine Learning Classification**









Project Overview**

This project is an introductory Machine Learning project focused on the classification of Iris flower species.

The objective is to build and evaluate several supervised Machine Learning models capable of predicting the species of an Iris flower from four numerical measurements.

Main objectives**

Explore a biological dataset

Prepare the data for Machine Learning

Split the data into training and testing sets

Train several classification models

Compare model performance

Apply cross-validation

Study overfitting

Analyze feature importance

Practice reproducible Machine Learning workflows

Publish the project on GitHub

Objective**

Given four measurements of an Iris flower:

Sepal length

Sepal width

Petal length

Petal width

the goal is to predict its species:

Iris setosa

Iris versicolor

Iris virginica

The complete workflow is:

**Dataset → EDA → Train/Test Split → Model Training → Evaluation → Cross-Validation → Overfitting Analysis → Interpretation**

Dataset**

This project uses the classic Iris dataset provided by scikit-learn.

Dataset characteristics**

| Property | Value |

|---|---:|

| Samples | 150 |

| Features | 4 |

| Classes | 3 |

| Missing values | None |

Features**

| Feature | Description |

|---|---|

| Sepal length | Length of the sepal in cm |

| Sepal width | Width of the sepal in cm |

| Petal length | Length of the petal in cm |

| Petal width | Width of the petal in cm |

Target classes**

| Label | Species |

|---:|---|

| 0 | Iris setosa |

| 1 | Iris versicolor |

| 2 | Iris virginica |

Exploratory Data Analysis**

The exploratory analysis includes:

Dataset inspection

Descriptive statistics

Feature distributions

Feature relationships

Class distribution

Visualization of the Iris species

The analysis shows that petal-related measurements provide strong separation between the three species.

Train/Test Split**

The dataset was divided into training and testing sets.

**80% training data:** 120 samples

**20% testing data:** 30 samples

The split was performed using:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

The test set was kept separate from model training and was used to evaluate performance on unseen samples.

Machine Learning Models

Four supervised classification algorithms were evaluated.

1. Logistic Regression

Logistic Regression was used as a baseline classification model.

It provides a simple and interpretable model for the classification problem.

2. Decision Tree

The Decision Tree learns decision rules from the input features.

It was also used to demonstrate the effect of model complexity and overfitting.

3. K-Nearest Neighbors

K-Nearest Neighbors (KNN) classifies a new observation according to the classes of its nearest neighbors.

The model was configured with:

n_neighbors = 5

4. Random Forest

Random Forest is an ensemble learning algorithm based on multiple decision trees.

The model was configured with:

n_estimators = 100

Model Evaluation

The models were evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion matrix

5-fold cross-validation

Accuracy

Accuracy represents the proportion of correctly classified samples:

Accuracy = Correct Predictions / Total Predictions

Precision

Precision measures how many observations predicted as a particular class actually belong to that class.

Recall

Recall measures how many observations belonging to a particular class were correctly identified.

F1-score

F1-score combines precision and recall into a single metric.

5-Fold Cross-Validation

A 5-fold cross-validation strategy was used to obtain a more robust estimate of model performance.

The dataset is divided into five folds:

Fold 1 → Validation
Fold 2 → Validation
Fold 3 → Validation
Fold 4 → Validation
Fold 5 → Validation

Each fold is used once for validation while the remaining folds are used for training.

The mean accuracy and standard deviation were calculated for each model.

This provides a more reliable estimate of model performance than relying on a single train/test split.

Overfitting Analysis

A Decision Tree was used to demonstrate overfitting.

Different tree depths were evaluated by comparing training and testing accuracy.

The general behavior is:

Low model complexity
        ↓
Underfitting

Optimal complexity
        ↓
Good generalization

High model complexity
        ↓
Overfitting

A highly complex model can achieve very high training accuracy while performing worse on unseen data.

This experiment demonstrates why model complexity must be controlled.

Feature Importance

Random Forest feature importance was used to investigate which features contributed most to the classification.

The four features were:

Sepal length

Sepal width

Petal length

Petal width

Petal-related measurements generally show high importance for distinguishing the Iris species.

Feature importance provides a simple introduction to Machine Learning model interpretability.

Results

The performance of the four models was compared using test-set accuracy and 5-fold cross-validation.

Model

Test Accuracy

Mean CV Accuracy

Logistic Regression

See notebook

See notebook

Decision Tree

See notebook

See notebook

K-Nearest Neighbors

See notebook

See notebook

Random Forest

See notebook

See notebook

The exact results can be found in:

notebooks/02_model_comparison.ipynb

The Iris dataset is small and relatively easy to classify. Therefore, several models can achieve high performance.

The main purpose of this project is to understand the complete Machine Learning workflow rather than simply maximizing accuracy.

Results and Figures

The project generates several figures.

Model Comparison



Comparison of the classification accuracy of the four Machine Learning models.

Feature Importance



Feature importance calculated using Random Forest.

Cross-Validation



Comparison of the mean accuracy obtained using 5-fold cross-validation.

Overfitting



Comparison of training and testing performance.

Decision Tree Complexity



Effect of Decision Tree depth on training and testing accuracy.

Project Structure

bioinformatics-ml-iris/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│
├── notebooks/
│   ├── 01_iris_exploration.ipynb
│   └── 02_model_comparison.ipynb
│
├── src/
│   └── train_model.py
│
└── results/
    └── figures/
        ├── model_comparison.png
        ├── feature_importance.png
        ├── cross_validation.png
        ├── overfitting.png
        └── decision_tree_overfitting.png

Technologies

Programming

Python 3.11

Data Analysis

NumPy

pandas

Visualization

Matplotlib

Seaborn

Machine Learning

scikit-learn

Development

Jupyter Notebook

Ubuntu / WSL

Git

GitHub

Installation

Clone the repository:

git clone https://github.com/Maha-bio/bioinformatics-ml-iris.git
cd bioinformatics-ml-iris

Create a Conda environment:

conda create -n iris-ml python=3.11 -y

Activate the environment:

conda activate iris-ml

Install the dependencies:

pip install -r requirements.txt

How to Run

Launch Jupyter Notebook:

jupyter notebook --no-browser

Open the Jupyter URL in your web browser.

Start with:

notebooks/01_iris_exploration.ipynb

Then run:

notebooks/02_model_comparison.ipynb

The notebooks cover the complete workflow from data exploration to Machine Learning model evaluation.

Key Machine Learning Concepts

This project provides practical experience with:

Supervised learning

Classification

Features and targets

Training and testing datasets

Model fitting

Prediction

Accuracy

Precision

Recall

F1-score

Confusion matrices

Cross-validation

Overfitting

Underfitting

Model complexity

Feature importance

Model comparison

Learning Outcomes

This project provided practical experience with the following Machine Learning workflow:

Problem Definition
        ↓
Data Exploration
        ↓
Data Preparation
        ↓
Train / Test Split
        ↓
Model Training
        ↓
Prediction
        ↓
Evaluation
        ↓
Cross-Validation
        ↓
Model Comparison
        ↓
Overfitting Analysis
        ↓
Interpretation

This project serves as a foundation for applying Machine Learning to more complex biomedical and bioinformatics datasets.

Limitations

Although the Iris dataset is useful for learning, it has several limitations:

Small number of samples

Only four features

Clean dataset

No missing values

Relatively easy classification problem

No high-dimensional biological features

Therefore, this project is primarily educational and should not be considered a realistic clinical prediction pipeline.

Future Improvements

Possible extensions include:

Hyperparameter tuning

GridSearchCV

RandomizedSearchCV

ROC curves

AUC comparison

Confusion matrix visualization

Model persistence using joblib

Prediction on new samples

Automated Machine Learning pipelines

Advanced model interpretation

Reproducible workflow automation

🧬 Next Step: Biomedical Machine Learning

This introductory project will serve as a foundation for applying Machine Learning to biomedical and bioinformatics datasets.

The next project will move toward a more realistic biomedical problem:

Biomedical Dataset
        ↓
Data Exploration
        ↓
Preprocessing
        ↓
Feature Selection
        ↓
Machine Learning
        ↓
Cross-Validation
        ↓
Model Evaluation
        ↓
Feature Interpretation
        ↓
Biological Interpretation

The long-term goal is to apply Machine Learning to high-dimensional biological data such as gene expression datasets.

Author

Maha Abbaci

Bioinformatics | Biomedical Data Analysis | Machine Learning

GitHub: Maha-bio