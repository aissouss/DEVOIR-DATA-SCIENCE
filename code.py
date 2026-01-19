"""
Analyse et Prédiction de la Survie du Titanic
==============================================
Devoir de Data Science - Licence 3 SI
Dataset : Titanic - Machine Learning from Disaster

Auteur : [Votre Nom]
Date : Janvier 2026
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Configuration des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# PARTIE 1 : CHARGEMENT ET COMPRÉHENSION DES DONNÉES
# =============================================================================
print("="*70)
print("PARTIE 1 : CHARGEMENT DES DONNÉES")
print("="*70)

# Charger le dataset
df = pd.read_csv("C:\\Users\\saiss\\Downloads\\titanic\\train.csv")

# Afficher les premières lignes
print("\nAperçu des données :")
print(df.head())

# Identifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# =============================================================================
# PARTIE 2 : NETTOYAGE ET PRÉPARATION DES DONNÉES
# =============================================================================
print("\n" + "="*70)
print("PARTIE 2 : NETTOYAGE DES DONNÉES")
print("="*70)

# Question 1 : Traitement des valeurs manquantes
print("\n[1] Traitement des valeurs manquantes...")

# Age : Imputation par la médiane
df['Age'] = df['Age'].fillna(df['Age'].median())
print(f"✓ Age : Rempli avec la médiane ({df['Age'].median():.2f} ans)")

# Cabin : Suppression de la colonne
df = df.drop('Cabin', axis=1)
print("✓ Cabin : Colonne supprimée (77% de valeurs manquantes)")

# Embarked : Imputation par le mode
mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode)
print(f"✓ Embarked : Rempli avec le mode ('{mode}')")

# Vérification
print(f"\n✓ Vérification : {df.isnull().sum().sum()} valeurs manquantes restantes")

# Question 2 : Encodage des variables catégorielles
print("\n[2] Encodage des variables catégorielles...")

# Sex : Encodage ordinal
encoder = OrdinalEncoder()
df[['Sex']] = encoder.fit_transform(df[['Sex']])
print("✓ Sex encodé (female=0, male=1)")

# Embarked : One-Hot Encoding
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', dtype=int)
print("✓ Embarked encodé (One-Hot : Embarked_C, Embarked_Q, Embarked_S)")

# Question 3 : Séparation X et y
print("\n[3] Séparation des variables...")

y = df['Survived']
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
        'Embarked_C', 'Embarked_Q', 'Embarked_S']]

print(f"✓ X : {X.shape[0]} lignes × {X.shape[1]} colonnes")
print(f"✓ y : {y.shape[0]} valeurs")

# =============================================================================
# PARTIE 3 : VISUALISATION DES DONNÉES
# =============================================================================
print("\n" + "="*70)
print("PARTIE 3 : VISUALISATION DES DONNÉES")
print("="*70)

# Question 1 : Distribution de la survie
print("\n[1] Distribution de la variable cible (Survived)...")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Survived')
plt.title('Distribution de la survie', fontsize=14, fontweight='bold')
plt.xlabel('Survie (0 = Décédé, 1 = Survécu)')
plt.ylabel('Nombre de passagers')
plt.tight_layout()
plt.show()

print(df['Survived'].value_counts())
print(f"Taux de survie : {df['Survived'].mean()*100:.2f}%")

# Question 2 : Survie selon le sexe
print("\n[2] Relation entre survie et sexe...")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survie selon le sexe', fontsize=14, fontweight='bold')
plt.xlabel('Sexe')
plt.ylabel('Nombre de passagers')
plt.legend(title='Survie', labels=['Décédé', 'Survécu'])
plt.xticks([0, 1], ['Femme', 'Homme'])
plt.tight_layout()
plt.show()

print("Taux de survie par sexe :")
print(df.groupby('Sex')['Survived'].mean() * 100)

# Question 3 : Survie selon la classe
print("\n[3] Impact de la classe sur la survie...")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survie selon la classe du billet', fontsize=14, fontweight='bold')
plt.xlabel('Classe (1 = Première, 2 = Deuxième, 3 = Troisième)')
plt.ylabel('Nombre de passagers')
plt.legend(title='Survie', labels=['Décédé', 'Survécu'])
plt.tight_layout()
plt.show()

print("Taux de survie par classe :")
print(df.groupby('Pclass')['Survived'].mean() * 100)

# Question 4 : Distribution de l'âge
print("\n[4] Distribution de l'âge...")
plt.figure(figsize=(8, 6))
plt.hist(df['Age'], bins=30, edgecolor='black', color='skyblue')
plt.title('Distribution de l\'âge des passagers', fontsize=14, fontweight='bold')
plt.xlabel('Âge')
plt.ylabel('Fréquence')
plt.tight_layout()
plt.show()

print("Statistiques de l'âge :")
print(df['Age'].describe())

# Question 5 : Visualisation supplémentaire
print("\n[5] Répartition par classe et sexe...")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Pclass', hue='Sex')
plt.title('Répartition des passagers par classe et sexe', fontsize=14, fontweight='bold')
plt.xlabel('Classe (1 = Première, 2 = Deuxième, 3 = Troisième)')
plt.ylabel('Nombre de passagers')
plt.legend(title='Sexe', labels=['Femme', 'Homme'])
plt.tight_layout()
plt.show()

print(df.groupby('Pclass')['Sex'].value_counts())

# =============================================================================
# PARTIE 4 : ANALYSE EXPLORATOIRE (EDA)
# =============================================================================
print("\n" + "="*70)
print("PARTIE 4 : ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
print("="*70)

# Question 1 : Statistiques descriptives
print("\n[1] Statistiques descriptives principales...")
print(df.describe())

# Question 2 : Analyse de corrélation
print("\n[2] Analyse de corrélation...")
correlation = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=1)
plt.title('Matrice de corrélation des variables numériques', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nCorrélations avec la survie :")
print(correlation['Survived'].sort_values(ascending=False))

# =============================================================================
# PARTIE 5 : MACHINE LEARNING SUPERVISÉ
# =============================================================================
print("\n" + "="*70)
print("PARTIE 5 : MACHINE LEARNING SUPERVISÉ")
print("="*70)

# Question 1 : Choix de l'algorithme
print("\n[1] Algorithme choisi : Régression Logistique")
print("Justification : Adapté à la classification binaire, interprétable, efficace.")

# Question 2 : Entraînement du modèle
print("\n[2] Entraînement du modèle...")

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
print(f"Ensemble d'entraînement : {X_train.shape[0]} observations")
print(f"Ensemble de test : {X_test.shape[0]} observations")

# Création et entraînement du modèle
logreg = LogisticRegression(random_state=16, max_iter=1000)
logreg.fit(X_train, y_train)
print("✓ Modèle entraîné")

# Prédictions
y_pred = logreg.predict(X_test)
print(f"\nExemple de prédictions (10 premières) : {y_pred[:10]}")
print(f"Valeurs réelles correspondantes : {y_test[:10].values}")

# Question 3 : Évaluation des performances
print("\n[3] Évaluation des performances...")

# Matrice de confusion
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :")
print(cnf_matrix)

# Métriques
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy:.2%}")
print(f"Precision : {precision:.2%}")
print(f"Recall : {recall:.2%}")

# =============================================================================
# PARTIE 6 : QUESTIONS DE RÉFLEXION
# =============================================================================
print("\n" + "="*70)
print("PARTIE 6 : CONCLUSION")
print("="*70)

print("""
LIMITES DU MODÈLE :
- Recall faible (63%) : manque 37% des survivants
- Pas de capture des interactions entre variables
- Hypothèse de linéarité restrictive
- Déséquilibre des classes (62% décès vs 38% survie)

AMÉLIORATIONS POSSIBLES :
- Créer des variables d'interaction (Sex × Pclass)
- Ingénierie de features (taille famille, infos Cabin)
- Tester Random Forest ou arbres de décision
- Équilibrer les classes (SMOTE)
- Validation croisée pour plus de robustesse
""")

print("\n" + "="*70)
print("ANALYSE TERMINÉE")
print("="*70)