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
git clone https://github.com/[votre-username]/titanic-analysis.git

# Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn

# Exécuter le notebook
jupyter notebook titanic_analysis.ipynb
```

## 📝 Structure du projet
```
titanic-analysis/
│
├── titanic_analysis.py      # Script principal
├── train.csv                 # Dataset
├── README.md                 # Documentation
└── requirements.txt          # Dépendances
```

## 👤 Auteur

[Votre Nom] - Licence 3 SI - [Date]

## 📄 Licence

Ce projet est réalisé dans un cadre académique.
```

---

## **requirements.txt**
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0