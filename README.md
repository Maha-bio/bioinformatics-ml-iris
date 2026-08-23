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