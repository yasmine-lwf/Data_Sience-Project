import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import RocCurveDisplay


#1-dataset
'''Credit Card Fraud:'''
#2-Analyser des données
#charger les donnees
df = pd.read_excel(r'C:\Users\ABL\OneDrive\Documentos\data2-chalnge3-Credit Card Fraud.xlsx')
print(df.head())

print(df.info())   
print(df.shape)
print(df.columns)
print(df.describe())
print(df.isnull().sum())  #les valeurs manquant 
print(df['fraud'].value_counts(normalize=True))  #pourcentage de variabe fraud 

'''on a obtenu : 0 → 91%, 1 → 8%,ce qui implique 'fraud' est  très déséquilibré.
 '''

#3-Gérer le déséquilibre des classes


# Séparer les classes
fraud = df[df.fraud == 1]
non_fraud = df[df.fraud == 0]

# Équilibrer par undersampling
non_fraud_downsampled = resample(non_fraud,
                                 replace=False,
                                 n_samples=len(fraud),
                                 random_state=42)

# Fusionner
df_balanced = pd.concat([fraud, non_fraud_downsampled])


# 4- Préparer les données
y=df_balanced['fraud']   #variable cible
x= df_balanced.drop("fraud", axis=1)   #variable explicative

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # Diviser les données : test and train



#5- modèle adapté
model = RandomForestClassifier(random_state=42)  # Création d'une instance du modèle
model.fit(x_train, y_train)  #Entraîner le modèle(Apprentissage du modèle)



#6- Évaluer la performance
y_pred = model.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#7-  Visualiser les résultats
# Matrice de confusion
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Matrice de confusion")
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(model, x_test, y_test)
plt.title("Courbe ROC")
plt.show()

# Importance des variables
importances = model.feature_importances_
feat_names = x.columns
sns.barplot(x=importances, y=feat_names)
plt.title("Importance des variables")
plt.show()


#8- conclusions
'''Le modele Random forest a bien performe sur le dataset de fraude par carte de credit '''


