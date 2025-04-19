import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#--------------------part1----------------------------


#1-charger les donnees 
donnes= pd.read_excel(r'C:\Users\ABL\OneDrive\Documentos\data1-projet-DS-.xlsx')
print(donnes)   # pour afficher le fichier excel 

print(donnes.info())   # Types de données, colonnes, valeurs manquantes
print(donnes.shape)       # (lignes, colonnes)                             
print(donnes.columns)     # Liste des colonnes

print(donnes.describe())  # Moyenne, écart-type, min, max, etc.



#2-verfier les valeur manquant (importante):
print(donnes.isnull().sum())  # Total des NaN par colonne -colonne vide-



#3-Explorer les colonnes uniques / catégories
for col in donnes.select_dtypes(include='object'):  # pour les colonnes catégorielles
    print(f"{col} - valeurs uniques :")
    print(donnes[col].value_counts())
    print()


#4-Visualisation rapide-graphes -plt-
'''----a------
#graphe distribution des valeur  -j'utilise sns. (bib seaborn) pour chaque colonne , 
# #sns.histplot(donnes['age'],bins=30 ,kde= True)
# plt.title('distribution d age ')
# plt.show()'''

'''----b------
#diagramme dispresion pour chaque pair de colonne on fait comme ca 
# sns.scatterplot(x='age', y='children'  , data=donnes)
# plt.title('diagramme dispresion ')
# plt.show() '''

#je peux faire directement pour tous les colonne a la foi comme ca (dispression + distribution):

# voila le code direct pour tous les colonne :

sns.pairplot(donnes.select_dtypes(include='number'))
plt.show()


#Matrice de corrélation :permet de voir les relations  entre toutes les variables numériques de DF


plt.figure(figsize=(10, 8)) 
matrice_corr=donnes.corr(numeric_only=True) 
'''.corr() : calcule la matrice de corrélation , (include='number') : sélectionne uniquement les colonnes numériques'''
#heatmap
sns.heatmap(matrice_corr,annot=True , cmap='coolwarm')
'''-heatmap :  une carte de chaleur (heatmap)
   -annot=True : affiche les valeurs numériques 
   -cmap='coolwarm' : choisit la palette de couleurs chaud/froid ie rouge , blanc ,blue
   -on peut ajouter auddi .2f pour prendre 2 nmbr apres la virgule'''
plt.title('matrice de corrélation')
plt.tight_layout()   #évite que le texte se chevauche
plt.show()
'''rmrq charges augmentent légèrement avec l’âge et un peu avec le BMI.
aucun relation entre childreen (nmbr enfant) et les variable :(charges , BMI et age) ,coorelation faible '''





















