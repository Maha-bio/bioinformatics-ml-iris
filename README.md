# 🧬 Bioinformatics AI/ML – Iris Classification


## 📌 Description


Ce projet est une introduction pratique à l'utilisation de l'**intelligence artificielle (IA)** et du **machine learning (ML)** en bioinformatique.


L'objectif est de construire un premier pipeline de classification permettant de comprendre les principales étapes d'un projet de Machine Learning :


- préparation des données
- exploration des données
- séparation TRAIN / TEST
- entraînement d'un modèle
- prédiction
- évaluation des performances
- visualisation des résultats
- sauvegarde du modèle


Le projet utilise le célèbre **Iris Dataset**, un jeu de données simple et largement utilisé pour apprendre la classification supervisée.


---


## 🎯 Objectifs du projet


À travers ce projet, nous allons apprendre à :


1. Charger un jeu de données
2. Explorer et comprendre les données
3. Préparer les variables pour le Machine Learning
4. Séparer les données en TRAIN et TEST
5. Entraîner un premier modèle de classification
6. Effectuer des prédictions
7. Évaluer les performances du modèle
8. Visualiser les résultats
9. Sauvegarder le modèle
10. Organiser un projet ML de manière reproductible


---


## 🧪 Dataset


Le projet utilise le **Iris Dataset**.


Le dataset contient trois espèces de fleurs :


- `setosa`
- `versicolor`
- `virginica`


Chaque observation contient quatre caractéristiques :


| Feature | Description |
|---|---|
| `sepal_length` | Longueur du sépale |
| `sepal_width` | Largeur du sépale |
| `petal_length` | Longueur du pétale |
| `petal_width` | Largeur du pétale |


La variable cible correspond à l'espèce de la fleur.


---


## 🧠 Machine Learning


Le problème est un problème de **classification supervisée**.


### Variables d'entrée


```text
sepal_length
sepal_width
petal_length
petal_width
Variable cible
species
Modèle utilisé

Le premier modèle utilisé dans ce projet est :

Logistic Regression

D'autres modèles pourront ensuite être ajoutés :

Decision Tree
Random Forest
Support Vector Machine
K-Nearest Neighbors
Gradient Boosting
📂 Structure du projet
iris-ai-ml/
│
├── README.md
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
├── config/
│   └── config.yaml
│
├── environment.yml
│
└── .gitignore
⚙️ Technologies utilisées
Programmation
Python
Bash
Data Science
NumPy
Pandas
Matplotlib
Seaborn
Machine Learning
Scikit-learn
Environnement
Conda
Jupyter Notebook
VS Code
Git
GitHub
🔧 Installation
1. Cloner le repository
git clone https://github.com/Maha-bio/iris-ai-ml.git
cd iris-ai-ml
2. Créer l'environnement Conda
conda env create -f environment.yml
3. Activer l'environnement
conda activate iris-ml
4. Lancer Jupyter Notebook
jupyter notebook
📊 Workflow du projet

Le pipeline suit les étapes suivantes :

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
TRAIN            TEST
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
🧩 Étape 1 – Chargement des données

Le dataset est chargé avec pandas ou directement depuis scikit-learn.

Exemple :

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
🧩 Étape 2 – Séparation TRAIN / TEST

Les données sont séparées en deux parties :

80 % TRAIN
20 % TEST
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

Le paramètre random_state=42 permet d'obtenir des résultats reproductibles.

🧩 Étape 3 – Entraînement du modèle

Le premier modèle utilisé est une régression logistique.

from sklearn.linear_model import LogisticRegression


model = LogisticRegression(
    max_iter=200
)


model.fit(X_train, y_train)
🧩 Étape 4 – Prédiction

Le modèle peut maintenant prédire les classes du jeu de test.

y_pred = model.predict(X_test)


print(y_pred)
🧩 Étape 5 – Évaluation

Plusieurs métriques sont utilisées pour évaluer le modèle.

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
📈 Métriques

Les principales métriques utilisées sont :

Accuracy

Mesure la proportion de prédictions correctes.

Accuracy = nombre de prédictions correctes / nombre total de prédictions
Precision

Mesure la proportion des prédictions positives qui sont réellement positives.

Recall

Mesure la capacité du modèle à identifier correctement les observations d'une classe.

F1-score

Combine la précision et le rappel.

📊 Visualisation

Les données peuvent être visualisées afin de comprendre la distribution des différentes espèces.

Exemple :

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
💾 Sauvegarde du modèle

Le modèle entraîné peut être sauvegardé afin d'être réutilisé sans devoir refaire l'entraînement.

import joblib


joblib.dump(
    model,
    "models/iris_model.pkl"
)

Pour charger le modèle :

model = joblib.load(
    "models/iris_model.pkl"
)
🔬 Pourquoi ce projet est intéressant en bioinformatique ?

Même si l'Iris Dataset est un exemple simple, le workflow utilisé ici correspond aux grandes étapes que l'on retrouve dans de nombreux projets de bioinformatique et de biologie computationnelle.

Par exemple :

Données biologiques
        │
        ▼
Prétraitement
        │
        ▼
Extraction des caractéristiques
        │
        ▼
TRAIN / TEST
        │
        ▼
Machine Learning
        │
        ▼
Prédictions
        │
        ▼
Évaluation
        │
        ▼
Interprétation biologique

Dans un projet réel, les caractéristiques pourraient être :

expression génique
variants génétiques
signatures moléculaires
données transcriptomiques
données protéomiques
données de méthylation
caractéristiques issues du cfDNA
données single-cell
🚀 Prochaines étapes

Le projet sera progressivement amélioré avec :

 Exploration complète du dataset
 Visualisation des données
 Normalisation des données
 Logistic Regression
 Decision Tree
 Random Forest
 SVM
 KNN
 Comparaison des modèles
 Cross-validation
 Hyperparameter tuning
 Matrice de confusion
 ROC curve
 AUC
 Sauvegarde du meilleur modèle
 Pipeline Scikit-learn
 Automatisation avec Snakemake
 Documentation complète
 Application du workflow à un dataset biologique réel
🧬 Vers un projet de bioinformatique réel

L'objectif final est de passer progressivement d'un problème pédagogique simple :

Iris Classification

à un problème de Machine Learning appliqué à la bioinformatique :

Biological Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning
        │
        ▼
Biological Prediction
        │
        ▼
Model Interpretation

Ce projet constitue donc une première étape vers le développement de pipelines reproductibles d'IA/ML appliqués aux données biologiques.

👩‍💻 Auteur

Maha-bio

Bioinformatics | Data Science | AI/ML | Computational Biology