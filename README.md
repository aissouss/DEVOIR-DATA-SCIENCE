# 🚢 Analyse et Prédiction de la Survie du Titanic

Projet d'analyse de données et de Machine Learning sur le dataset Titanic.

## 📋 Description

Ce projet analyse les facteurs qui ont influencé la survie des passagers du Titanic en utilisant des techniques de Data Science et de Machine Learning.

## 🎯 Objectifs

- Nettoyer et préparer les données (traitement des valeurs manquantes, encodage)
- Visualiser les relations entre variables
- Analyser les corrélations
- Prédire la survie avec un modèle de Régression Logistique

## 📊 Dataset

**Source** : [Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic)

**Variables principales** :
- `Survived` : Survie (0 = Non, 1 = Oui)
- `Pclass` : Classe du billet (1, 2, 3)
- `Sex` : Sexe
- `Age` : Âge
- `Fare` : Prix du billet
- `Embarked` : Port d'embarquement

## 🛠️ Technologies utilisées

- **Python 3.x**
- **Pandas** : Manipulation de données
- **Matplotlib & Seaborn** : Visualisation
- **Scikit-learn** : Machine Learning

## 📈 Résultats

- **Accuracy** : 77.13%
- **Precision** : 73.97%
- **Recall** : 62.79%

**Observations clés** :
- Le sexe et la classe sont les facteurs les plus déterminants
- Les femmes ont 4× plus de chances de survie que les hommes
- La 1ère classe a 2.6× plus de chances que la 3ème classe

## 🚀 Installation et exécution
```bash
# Cloner le repository
git clone https://github.com/aissouss/DEVOIR-DATA-SCIENCE.git

# Accéder au dossier
cd DEVOIR-DATA-SCIENCE

# Installer les dépendances
pip install -r requirements.txt

# Exécuter le script
python code.py
```

## 📝 Structure du projet
```
DEVOIR-DATA-SCIENCE/
│
├── code.py                   # Script principal d'analyse
├── train.csv                 # Dataset Titanic
├── README.md                 # Documentation du projet
└── requirements.txt          # Dépendances Python
```

## 👤 Auteur

**AISSYA BOUKRAA** - Licence 3 SI - Janvier 2026

## 📄 Licence

Ce projet est réalisé dans un cadre académique.
