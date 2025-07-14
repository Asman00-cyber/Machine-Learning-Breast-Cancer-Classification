import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler , MinMaxScaler
from sklearn import svm 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold , GridSearchCV
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest , f_classif
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import pandas as pd
import random

###################################### DATA LOAD/PREPARATION/EDA ###############################################################
SEED=np.random.randint(0,10e7)
print(f"SEED: {SEED}")
random_state = SEED

df = pd.read_csv(r'./datasets/breast-cancer.csv')
print(df.shape)

#are thery are NaN or null values in the data set? If yes, extract them 
print(f"Are there any NaN values?-> {df.isna().values.any()}")
print(f"Are there any null values?-> {df.isna().values.any()}")

#split the features and the classes into two subsets

#drop the id column, we dont actually need it:
df.drop(labels= 'id' , axis = 1 , inplace=True)
#check if id columns is actually dropped
print(df.head())
print(df.shape)

#X-> all the features , y ->classes column
X = df.iloc[: , 1:]
y = df.iloc[: , 0]

#set the problem into a binary classification problem by putting M=1 and B=0
le = LabelEncoder()
y=le.fit_transform(y)

#Histograms of all the features
labels = []
for label in X.columns:
    labels.append(label)

def plot_hist(X , bins):
    fig , axes = plt.subplots(5,6 , figsize=(15,15))
    axes =axes.flatten()
    for i, label in enumerate(labels):
        axes[i].hist(X[label] , bins =bins , edgecolor ='black')
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel(f"Count {label} ")
        
    plt.tight_layout()
    plt.show()
plot_hist(X,30)

#Plot boxplots for each feature

def box_plot(X):
    fig , axes = plt.subplots(5,6 , figsize = (15,15))
    axes = axes.flatten()
    for i, label in enumerate(labels):
        axes[i].boxplot(X[label] ,orientation = 'vertical' , notch = False )
        axes[i].set_title(f"{label}")
    plt.tight_layout()
    plt.show()
box_plot(X)

    
#Correlation Heatmap for all the features of the dataset
p_corr = X.corr()

plt.figure(figsize=(8,8))
sns.heatmap(p_corr ,cmap= 'crest' , linewidths=0.1 , linecolor='b' , xticklabels=True , yticklabels=True)
plt.title("Correlation heatmap for the dataset features")
plt.show()


#Split the X and y into train and test subsets for furhter analysis
X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size=0.2 , train_size=0.8  ,random_state=random_state)

#I will use the KNN classifier which is based on distances between data points.So the values need to be scaled(only for the KNN)
scl = StandardScaler()
X_train_scaled = scl.fit_transform(X_train)
X_test_scaled = scl.transform(X_test)



##################################### DATA ANALYSIS/MODELING ##################################################################

#On this part i will use different models to see which one has the best accuracy for this dataset
#KNN classifier 

#Form a pipeline (feature selection using KBest)

pipe1 = Pipeline(steps=[
    ("feature selection", SelectKBest(score_func=f_classif , k=20)),
    ("Classifier" , KNeighborsClassifier(n_neighbors=20 , weights='distance' , algorithm='ball_tree' ,metric ="minkowski" , p=1))
], verbose=True)

def get_best_features(X_train=X_train_scaled , y_train=y_train):
    pipe1.fit(X_train_scaled , y_train)
    feature_selector = pipe1.named_steps["feature selection"]
    selected_features = feature_selector.get_feature_names_out(X.columns)
    return selected_features


best_features = get_best_features()
print(f"Best features selected for KNN model:{best_features}")

#Use the only the selected features for the KNN model evaluation

X_train_selected = pipe1.named_steps['feature selection'].transform(X_train_scaled)  
X_test_selected = pipe1.named_steps['feature selection'].transform(X_test_scaled) 

training_accuracy = pipe1.named_steps["Classifier"].score(X_train_selected , y_train)
test_accuracy = pipe1.named_steps["Classifier"].score(X_test_selected , y_test)
print(f"Training accuracy:{training_accuracy}")
print(f"Testing accuracy:{test_accuracy}")




