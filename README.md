

# Bioinformatics AI/ML – Iris Classification

## Description

This project is a practical introduction to the use of **Artificial Intelligence (AI)** and **Machine Learning (ML)** in bioinformatics.

The goal is to build a first classification pipeline to understand the main steps of a Machine Learning project:

- Data preparation
- Data exploration
- TRAIN / TEST splitting
- Model training
- Prediction
- Performance evaluation
- Results visualization
- Model comparison
- Model saving
- Project reproducibility

The project uses the well-known **Iris Dataset**, a simple dataset widely used to learn the fundamentals of supervised classification.

---

## Project Objectives

Through this project, we will learn how to:

1. Load a dataset
2. Explore and understand the data
3. Prepare variables for Machine Learning
4. Split the data into **TRAIN** and **TEST** sets
5. Train different classification models
6. Make predictions
7. Evaluate model performance
8. Compare different models
9. Visualize the results
10. Save the best-performing model
11. Organize a reproducible Machine Learning project

---

## Dataset

The project uses the **Iris Dataset**.

### Species

The dataset contains three flower species:

- `setosa`
- `versicolor`
- `virginica`

### Features

Each observation contains four features:

| Feature | Description |
|---|---|
| `sepal_length` | Sepal length |
| `sepal_width` | Sepal width |
| `petal_length` | Petal length |
| `petal_width` | Petal width |

The target variable corresponds to the flower species.

---

## Machine Learning

This project addresses a **supervised classification** problem.

### Input Variables

```text
sepal_length
sepal_width
petal_length
petal_width
```

### Target Variable

```text
species
```

### Models Studied

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

---

## Project Structure

```text
iris-ai-ml/
│
├── README.md
├── environment.yml
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── iris_classification.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── models/
│   └── iris_model.pkl
│
├── results/
│   ├── figures/
│   └── metrics/
│
└── config/
    └── config.yaml
```

---

## Technologies Used

### Programming

- Python
- Bash

### Data Science

- NumPy
- Pandas
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Environment and Tools

- Conda
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Maha-bio/bioinformatics-ml-iris.git
cd bioinformatics-ml-iris
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
```

### 3. Activate the Environment

```bash
conda activate iris-ml
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## Project Workflow

The pipeline follows the main steps of a Machine Learning project:

```text
Dataset
   │
   ▼
Data Exploration
   │
   ▼
Data Preprocessing
   │
   ▼
TRAIN / TEST Split
   │
   ├──────────────┐
   ▼              ▼
 TRAIN           TEST
   │              │
   ▼              │
Model Training    │
   │              │
   ▼              │
Predictions ◄─────┘
   │
   ▼
Evaluation
   │
   ▼
Results
```

---

# Project Steps

## 1. Loading the Data

The dataset is loaded directly from `scikit-learn`.

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(
    iris.target,
    name="species"
)

print(X.head())
print(y.head())
```

---

## 2. TRAIN / TEST Split

The data is divided into two sets:

- **80% for training**
- **20% for testing**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)
```

The `random_state=42` parameter ensures reproducible results.

The `stratify=y` parameter maintains a similar class distribution in the TRAIN and TEST sets.

---

## 3. Model Training

The first model used is **Logistic Regression**.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=200
)

model.fit(X_train, y_train)
```

---

## 4. Prediction

Once the model has been trained, it can predict the classes of the test set.

```python
y_pred = model.predict(X_test)

print(y_pred)
```

---

## 5. Model Evaluation

Several metrics are used to evaluate model performance.

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
```

---

# Evaluation Metrics

## Accuracy

Accuracy measures the proportion of correctly classified predictions.

```text
Accuracy =
Number of correct predictions
-----------------------------
Total number of predictions
```

## Precision

**Precision** measures the proportion of positive predictions that are actually positive.

## Recall

**Recall** measures the ability of the model to correctly identify observations belonging to a specific class.

## F1-score

The **F1-score** combines precision and recall to provide an overall measure of model performance.

---

# Data Visualization

The data can be visualized to better understand the distribution of the different flower species.

### Example

```python
import matplotlib.pyplot as plt

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y
)

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris Dataset")

plt.show()
```

---

# Model Saving

The trained model can be saved so that it can be reused later without retraining.

```python
import joblib

joblib.dump(
    model,
    "models/iris_model.pkl"
)
```

### Loading the Model

```python
model = joblib.load(
    "models/iris_model.pkl"
)
```

---



In a real bioinformatics project, Machine Learning features could include:

- Gene expression
- Genetic variants
- Molecular signatures
- Transcriptomic data
- Proteomic data
- DNA methylation data
- cfDNA-derived features
- Single-cell data

---


---

# Towards a Real Bioinformatics Project

The ultimate goal is to progressively move from a simple educational problem:

```text
Iris Classification
```



This project represents a first step toward developing **reproducible AI/ML pipelines for biological data analysis**.

---

# Expected Results

At the end of the project, we will have:

- A properly prepared dataset
- A TRAIN / TEST split
- Several classification models
- Model performance metrics
- Data visualizations
- A saved trained model
- A reproducible project structure
- A foundation for developing Machine Learning projects in bioinformatics

---

#  Author

**Maha Abbaci**

*Bioinformatics | Computational Biology | AI/ML*

GitHub: https://github.com/Maha-bio

---

# License

This project is intended for **academic, research, and educational purposes**.


#  Bioinformatics AI/ML – Iris Classification

##  Description

Ce projet est une introduction pratique à l'utilisation de l'**intelligence artificielle (IA)** et du **Machine Learning (ML)** en bioinformatique.

L'objectif est de construire un premier pipeline de classification permettant de comprendre les principales étapes d'un projet de Machine Learning :

- Préparation des données
- Exploration des données
- Séparation des données en **TRAIN / TEST**
- Entraînement des modèles
- Prédiction
- Évaluation des performances
- Visualisation des résultats
- Comparaison des modèles
- Sauvegarde du modèle
- Reproductibilité du projet

Le projet utilise le célèbre **Iris Dataset**, un jeu de données simple et largement utilisé pour apprendre les concepts fondamentaux de la classification supervisée.

---

##  Objectifs du projet

À travers ce projet, nous allons apprendre à :

1. Charger un jeu de données
2. Explorer et comprendre les données
3. Préparer les variables pour le Machine Learning
4. Séparer les données en ensembles **TRAIN** et **TEST**
5. Entraîner différents modèles de classification
6. Effectuer des prédictions
7. Évaluer les performances des modèles
8. Comparer les modèles
9. Visualiser les résultats
10. Sauvegarder le meilleur modèle
11. Organiser un projet ML de manière reproductible

---

##  Dataset

Le projet utilise le **Iris Dataset**.

###  Espèces

Le dataset contient trois espèces de fleurs :

- `setosa`
- `versicolor`
- `virginica`

###  Caractéristiques

Chaque observation contient quatre caractéristiques :

| Feature | Description |
|---|---|
| `sepal_length` | Longueur du sépale |
| `sepal_width` | Largeur du sépale |
| `petal_length` | Longueur du pétale |
| `petal_width` | Largeur du pétale |

La variable cible correspond à l'espèce de la fleur.

---

##  Machine Learning

Le problème traité est un problème de **classification supervisée**.

### Variables d'entrée

```text
sepal_length
sepal_width
petal_length
petal_width
```

### Variable cible

```text
species
```

### Modèles étudiés

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

---

##  Structure du projet

```text
iris-ai-ml/
│
├── README.md
├── environment.yml
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── iris_classification.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── models/
│   └── iris_model.pkl
│
├── results/
│   ├── figures/
│   └── metrics/
│
└── config/
    └── config.yaml
```

---

##  Technologies utilisées

###  Programmation

- Python
- Bash

###  Data Science

- NumPy
- Pandas
- Matplotlib
- Seaborn

###  Machine Learning

- Scikit-learn

###  Environnement et outils

- Conda
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

##  Installation

### 1. Cloner le repository

```bash
git clone https://github.com/Maha-bio/bioinformatics-ml-iris.git
cd bioinformatics-ml-iris
```

### 2. Créer l'environnement Conda

```bash
conda env create -f environment.yml
```

### 3. Activer l'environnement

```bash
conda activate iris-ml
```

### 4. Lancer Jupyter Notebook

```bash
jupyter notebook
```

---

##  Workflow du projet

Le pipeline suit les principales étapes d'un projet de Machine Learning :

```text
Dataset
   │
   ▼
Data Exploration
   │
   ▼
Data Preprocessing
   │
   ▼
TRAIN / TEST Split
   │
   ├──────────────┐
   ▼              ▼
 TRAIN           TEST
   │              │
   ▼              │
Model Training    │
   │              │
   ▼              │
Predictions ◄─────┘
   │
   ▼
Evaluation
   │
   ▼
Results
```

---

#  Étapes du projet

## 1. Chargement des données

Le dataset est chargé directement à partir de `scikit-learn`.

```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(
    iris.target,
    name="species"
)

print(X.head())
print(y.head())
```

---

## 2. Séparation TRAIN / TEST

Les données sont séparées en deux ensembles :

- **80 % pour l'entraînement**
- **20 % pour le test**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)
```

Le paramètre `random_state=42` permet d'obtenir des résultats reproductibles.

Le paramètre `stratify=y` permet de conserver une répartition similaire des classes dans les ensembles TRAIN et TEST.

---

## 3. Entraînement du modèle

Le premier modèle utilisé est une **régression logistique**.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=200
)

model.fit(X_train, y_train)
```

---

## 4. Prédiction

Une fois le modèle entraîné, il peut prédire les classes du jeu de test.

```python
y_pred = model.predict(X_test)

print(y_pred)
```

---

## 5. Évaluation

Plusieurs métriques sont utilisées pour évaluer les performances du modèle.

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
```

---

#  Métriques d'évaluation

## Accuracy

L'accuracy mesure la proportion de prédictions correctement classées.

```text
Accuracy =
Nombre de prédictions correctes
--------------------------------
Nombre total de prédictions
```

## Precision

La **precision** mesure la proportion des prédictions positives qui sont réellement positives.

## Recall

Le **recall** mesure la capacité du modèle à identifier correctement les observations appartenant à une classe.

## F1-score

Le **F1-score** combine la precision et le recall afin de fournir une mesure globale des performances du modèle.

---

#  Visualisation des données

Les données peuvent être visualisées afin de mieux comprendre la distribution des différentes espèces.

### Exemple

```python
import matplotlib.pyplot as plt

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y
)

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris Dataset")

plt.show()
```

---

#  Sauvegarde du modèle

Le modèle entraîné peut être sauvegardé afin d'être réutilisé ultérieurement sans devoir refaire l'entraînement.

```python
import joblib

joblib.dump(
    model,
    "models/iris_model.pkl"
)
```

### Charger le modèle

```python
model = joblib.load(
    "models/iris_model.pkl"
)
```



Dans un projet de bioinformatique réel, les caractéristiques utilisées pour le Machine Learning pourraient être :

- Expression génique
- Variants génétiques
- Signatures moléculaires
- Données transcriptomiques
- Données protéomiques
- Données de méthylation
- Caractéristiques issues du cfDNA
- Données single-cell

---


---

# Résultats attendus

À la fin du projet, nous disposerons de :

- Un dataset correctement préparé
- Un ensemble TRAIN / TEST
- Plusieurs modèles de classification
- Des métriques de performance
- Des visualisations
- Un modèle entraîné sauvegardé
- Une structure de projet reproductible
- Une base pour développer des projets de Machine Learning en bioinformatique

---

# Auteur

**Maha Abbaci**

*Bioinformatics | Computational Biology| AI/ML *

GitHub: https://github.com/Maha-bio

---

# Licence

This project is intended for academic, research and educational purposes.

