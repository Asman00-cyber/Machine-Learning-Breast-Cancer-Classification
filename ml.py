import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn import svm 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold , GridSearchCV
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest , f_classif
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import random

###################################### DATA LOAD AND PREPARATION/PREPROCESSING #################################################
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


#Split the X and y into train and test subsets for furhter analysis
X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size=0.2 , train_size=0.8  ,random_state=random_state)

#I will use the KNN classifier which is based on distances between data points.So the values need to be scaled(only for the KNN)
scl = StandardScaler()
X_train_scaled = scl.fit_transform(X_train)
X_test_scaled = scl.transform(X_test)


################################### EXPLORATORY DATA ANALYSIS ############################################################################
#Histograms of specific features
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


#Correlation Heatmap for all the features of the dataset

X_train_scaled_data = pd.DataFrame(X_train_scaled , columns=X.columns) #transform X_train_scaled into a dataframe to use corr func
p_corr = X_train_scaled_data.corr()

plt.figure(figsize=(8,8))
sns.heatmap(p_corr ,cmap= 'crest' , linewidths=0.1 , linecolor='b' , xticklabels=True , yticklabels=True)
plt.title("Correlation heatmap for the dataset features")
plt.show()

##################################### DATA ANALYSIS/MODELING ##################################################################

#KNN classifier 
#Form a pipeline (feature selection inside GridSearch)

params ={'feature_selection__k': [10, 15, 20,25,30],  # How many features to select
    'classifier__n_neighbors': [10, 15, 20, 25, 30],          # KNN neighbors
    'classifier__weights': ['uniform', 'distance'],            # KNN weights
    'classifier__metric': ['minkowski'],                       # KNN metric
    'classifier__p': [1, 2]}                                    # p=1-> ManHattan , p=2->Euclidean

#Pipeline of feature selection for the KNN classifier
pipeline = Pipeline([('feature_selection' , SelectKBest(score_func=f_classif)) , ('classifier' ,KNeighborsClassifier())])

#Gridsearch to slect the best features and best hyperparams for the model
grid_search = GridSearchCV(pipeline , param_grid=params , cv = KFold(n_splits=5 , shuffle=True,random_state=random_state) , scoring='accuracy' , verbose=1)
grid_search.fit(X_train_scaled , y_train)
grid_pred = grid_search.predict(X_test_scaled)


print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

#Get the selected features
best_model = grid_search.best_estimator_
feature_selector = best_model.named_steps['feature_selection']
selected_features = X.columns[feature_selector.get_support()]
print("Selected features:", list(selected_features))

# Get classifier parameters:
classifier = best_model.named_steps['classifier'] 
print(f"KNN: n_neighbors={classifier.n_neighbors}, weights={classifier.weights} , metric:{classifier.metric} , p:{classifier.p}")

