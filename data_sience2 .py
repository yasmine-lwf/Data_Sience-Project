import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


#1-Téléchargement de data set :
my_data = pd.read_excel(r'C:\Users\ABL\OneDrive\Documentos\data1-projet-DS-.xlsx')
print('le fichier excel :    ',my_data)

#2-Préparation des données :
'''2-1-la variable cible (charges) '''
y = my_data['charges']    # variable à prédire

'''2-2-le rest des colonne sont les  caractéristiques  'input' '''
x = my_data.drop(columns=['charges'])    # toutes les autres colonnes

'''2-3-Transformation des variables catégorielles en nombres .'''
my_data['sex']    = my_data['sex'].map({'male' :1 , 'female' :0})
my_data['smoker'] = my_data['smoker'].map({'yes':1 , 'no': 0})

my_data =pd.get_dummies(my_data , columns=['region'] , drop_first=True) 



#3-Diviser les données : test and train
x_train ,x_test , y_train , y_test =train_test_split(x,y ,test_size=0.2  ,  random_state=42)  #on peut ajouter random_state=42 → c’est une "graine aléatoire", juste pour que les mêmes données soient toujours utilisées quand tu relances le code (utile pour comparer)
''' 
-20% des données pour le test, 80% pour l'entraînement
-Donc avec 1337 lignes :
              80% train : 1069 lignes
              20% test :  268 lignes
'''


#4. Choisir un modèle : Classification ou régression
'''
- 'charges' est un nombre (float), donc  on utilise le model :régression.
   avec  :  LinearRegression()
'''
# Création d'une instance du modèle
model = LinearRegression()



#5- Entraîner le modèle(Apprentissage du modèle)
model.fit(x_train,y_train)
''' 
    X_train : les entrées (âge, sexe, ...)
    y_train : la sortie à prédire ( charges)    '''


#6-Tester et évaluer
'''objec: Vérifier si le modèle est bon, c'est-à-dire si ses prédictions sont proches de la réalité.
'''
y_pred=model.predict(x_test)

mse = mean_squared_error(y_test , y_pred)
'''-mean_squared_error(y_test, y_pred) : On compare les vraies valeurs (y_test) aux valeurs prédites (y_pred) pour calculer l'erreur (MSE = moyenne des carrés des erreurs).
Plus mse est petit, c'est  mieux 
'''

r2 = r2_score(y_test,y_pred)
'''
   -r2_score = pourcentage de la variance expliquée par le modèle,varie entre 0 et 1.
'''

'''Afficher les résultats de l'évaluation.'''
print("Mean Squared Error:", mse)
print("R2 Score:", r2)



#7-Comparer plusieurs modèles
#Une liste de tuples : chaque élément contient un nom (string) et un modèle (instance).
models = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor()),
    ("Random Forest", RandomForestRegressor())
]

# Entraîner chaque modèle + l'évaluer
for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE={mse:.2f}, R2={r2:.2f}")



#8-CONCLUSION CLAIRE À ÉCRIRE
'''  
-Après avoir testé trois modèles de régression (régression linéaire, arbre de décision et forêt aléatoire), le modèle 'Random Forest Regressor' est le plus performant avec  [R² = 0.86]  (le plus élevé) et [MSE = 21.2M ] (le plus faible).
donc qu'il est capable d'expliquer 86% de la variation des charges , tout en ayant des erreurs de prédiction moins importantes que les autres modèles.
-Conclusion : Le modèle de forêt aléatoire ['Random Forest Regressor' ] est le plus adapté pour ce problème de prédiction de charges.

'''