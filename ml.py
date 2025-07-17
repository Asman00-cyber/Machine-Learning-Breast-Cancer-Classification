import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler , MinMaxScaler
from sklearn.svm import SVC #SVM library
from sklearn.inspection import DecisionBoundaryDisplay #library for SVM boundary display
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold , learning_curve , cross_val_score , StratifiedKFold
from sklearn.feature_selection import SelectKBest , f_classif
from sklearn.metrics import ConfusionMatrixDisplay ,confusion_matrix , accuracy_score , precision_score , recall_score , f1_score
import seaborn as sns
import pandas as pd
import random

###################################### DATA LOAD/PREPARATION/EDA ###############################################################
# SEED=np.random.randint(0,10e7)
# print(f"SEED: {SEED}")
random_state = 22457458

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
#In this case i have 569 samples and 31 features. By the rule of thumb 569samples/31 features = 18 samples per feature. This means i have
#high risk of overfitting if i DONT apply feature selection. By increasing the samples per feature i reduce the overfitting propabilities,thats why 
#i chose to reduce the features to do the analysis!

pipe1 = Pipeline(steps=[
    ("feature selection", SelectKBest(score_func=f_classif , k=20)),
    ("Classifier" , KNeighborsClassifier())
], verbose=True)


#get the best features selected
def get_best_features(X_train=X_train_scaled , y_train=y_train):
    pipe1.fit(X_train_scaled , y_train)
    feature_selector = pipe1.named_steps["feature selection"]
    selected_features = feature_selector.get_feature_names_out(X.columns)
    return selected_features

best_features = get_best_features()
print(f"Best features selected for KNN model:{best_features}")
X_train_selected = pipe1.named_steps['feature selection'].transform(X_train_scaled)  
X_test_selected = pipe1.named_steps['feature selection'].transform(X_test_scaled)

#print the shape of the X_train_selected and X_test selected for debugging purposes
print(f"Shape of the X_train_selected:{X_train_selected.shape}")
print(f"Shape of the X_test_selected:{X_test_selected.shape}")

#get the training curve to see if the model has more space to be trained(if it requires more data)
def get_training_curve(X_train =X_train_scaled, y_train =y_train):
    X_train_size , X_train_scores ,X_test_scores = learning_curve(pipe1 , X_train_selected , y_train , cv = KFold(n_splits=10),
                                                                  train_sizes=np.linspace(0.1 , 1.0, 10),verbose=True , n_jobs=-1,
                                                                  scoring="accuracy" , random_state=random_state , shuffle=True)
    x_train_scores_mean =np.mean(X_train_scores , axis =1)
    plt.plot(X_train_size , x_train_scores_mean , color = 'b' , linestyle='solid'  , label="Training proceess")
    plt.xlabel("Training examples")
    plt.ylabel("Training score")
    plt.legend()
    plt.grid(True)
    plt.show()
get_training_curve()

#by observing the training curve , we can see that the training score doesnt reach a plateau. The model keeps on "learning". This means that
#more data is required for the evaluation of the model


#Function that returns the confusion matrix of the KNN model
def get_confussion_matrix(y_test=y_test):
    y_pred =pipe1.named_steps['Classifier'].predict(X_test_selected)
    confusion_matr = confusion_matrix(y_test , y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matr )
    disp.plot()
    plt.title("Confussion Matrix for KNN model")
    plt.show()
get_confussion_matrix()


#Function that returns the confusion matrix metrics for the KNN model
def get_confusion_metrics(y_test=y_test):
    y_pred = pipe1.named_steps["Classifier"].predict(X_test_selected)
    accuracy_knn = accuracy_score(y_test ,y_pred)
    precision_knn = precision_score(y_test , y_pred)
    recall_knn = recall_score(y_test , y_pred)
    f1_score_knn = f1_score(y_test , y_pred)
    labels = ["Accuracy KNN" , "Precision KNN" , "Recal KNN" , "F1-score KNN"]
    metrics = [accuracy_knn ,precision_knn ,recall_knn , f1_score_knn]
    colors = ['red', 'blue' , 'green' , 'violet']
    fig , axes = plt.subplots(1,4 ,figsize = (8,8))
    axes =axes.flatten()
    axes[0].set_title("Confussion Matrix metrics for KNN")
    for i in range(len(labels)):
        axes[i].bar(labels[i] ,metrics[i] , color = colors[i])

    plt.tight_layout()
    plt.show()
get_confusion_metrics()

#print out the training accuracy and the test accuracy on the terminal for the KNN model
training_accuracy = pipe1.named_steps["Classifier"].score(X_train_selected , y_train)
test_accuracy = pipe1.named_steps["Classifier"].score(X_test_selected , y_test)
print(f"Training accuracy:{training_accuracy}")
print(f"Testing accuracy:{test_accuracy}")

#compute the cross_validation scores for 10 iteration on the KNN model to see how good it generilizes on unseen/dhuffled data
def cross_val_knn():
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    cv_scores = cross_val_score(pipe1, X_train_scaled, y_train, 
                               cv=cv, scoring='accuracy')
    
    print("cross-validation diagnosis KNN:")
    print(f"CV KNN scores: {cv_scores}")
    print(f"CV KNN scores mean: {cv_scores.mean()}")
cross_val_knn()


#SVM classifier

#for the SVM i will use different Kernels(linear,polynomial, rbf) in order to evaluate the perfomance of the model for the classification

#by observing the correlation heatmap of the EDA section, i will use two low correlated features and plot the SVM boundary plot
#to see how efficient is each SVM model when it comes to setting boundaries

#the selected features are radius_mean and smoothness_se 

def plot_SVM_boundaries(X_train =X_train_scaled , y_train =y_train):
    global X
    SVM_linear = SVC(kernel='linear' , C =1 , gamma= 'auto' , random_state=random_state)
    SVM_polynomial = SVC(kernel='poly', C=1 , gamma='auto' , degree=3 , random_state=random_state)
    SVM_rbf = SVC(kernel='rbf' , gamma ='auto', C=1 , random_state=random_state)
    SVM_list = [SVM_linear , SVM_polynomial , SVM_rbf]
    fig , axes = plt.subplots (1,3 ,figsize = (12,12))
    X_train_scaled_df= pd.DataFrame(X_train_scaled , columns =X.columns)  #convert the X_train_scaled into a datafram to use the .loc command
    titles = ("SVM linear" , "SVM polynomial" , "SVM rbf") #store the titles of the models in a list
    X1 = X_train_scaled_df['radius_mean'].values
    X2 = X_train_scaled_df['smoothness_se'].values
    X_ = np.column_stack([X1,X2])
    SVM_models = [clf.fit(X_ , y_train) for clf in SVM_list]

    for clf , title , ax in zip(SVM_models , titles , axes.flatten()):
        DecisionBoundaryDisplay.from_estimator(clf ,X_ , grid_resolution=100,
                                               response_method='predict' , cmap = plt.cm.coolwarm ,
                                                alpha = 0.8 , xlabel='radius_mean', ylabel='smoothness_se' , ax =ax)
        ax.scatter(X1 ,X2 , c=y_train, cmap = plt.cm.coolwarm , s=10 , edgecolors = 'k')
        ax.set_xlabel('radius mean')
        ax.set_ylabel('smoothness_se')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.grid(True)
    plt.tight_layout()
    plt.show()
plot_SVM_boundaries()
#By observing the three different SVM boundary plots , we can see that the best classification is occuring while using the RBF kernenl!


#I form a new pipeline called pipe2 for the SVM classifier respectively
#the pipeline will be of the same logic as followed for the KNN classifier
pipe2_rbf = Pipeline([("feature selection" , SelectKBest(k=20 , score_func=f_classif)),
                ("Classifier", SVC(kernel='rbf' , C =1 , random_state=random_state, gamma="auto"))])
pipe2_rbf.fit(X_train_scaled , y_train) #fit the pipeline to the training data


def get_training_curve_SVM(X_train =X_train_scaled , y_train =y_train):
    X_train_size ,X_train_score ,X_test_score = learning_curve(pipe2_rbf , X_train_scaled , y_train , 
                                                               train_sizes=np.linspace(0.1,1.0,10) , cv=KFold(n_splits=10),
                                                               scoring='accuracy' , verbose=True,random_state=random_state,n_jobs=-1)
    x_train_mean_per_iteration = np.mean(X_train_score , axis = 1)
    plt.plot(X_train_size , x_train_mean_per_iteration , color = 'b', linestyle = 'solid' )
    plt.grid(True)
    plt.xlabel("Training samples")
    plt.ylabel('Training score')
    plt.title('SVM training curve')
    plt.show()
get_training_curve_SVM()
#By observing the SVM training curve we can see there is more room for training, meaning the dataset is too small

def plot_confusion_matrix_SVM(y_test=y_test):
    y_pred_SVM = pipe2_rbf.predict(X_test_scaled)
    confusion_matr = confusion_matrix(y_test , y_pred_SVM)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matr)
    disp.plot()
    plt.title("Confusion Matrix for SVM")
    plt.show()
plot_confusion_matrix_SVM()


def plot_confusion_metrics_SVM(X_train = X_train_scaled , y_train = y_train , y_test = y_test , X_test =X_test_scaled):
    y_pred_SVM = pipe2_rbf.predict(X_test)
    accuracy_SVM = accuracy_score(y_test , y_pred_SVM)
    precision_SVM = precision_score(y_test , y_pred_SVM)
    recall_SVM = recall_score(y_test , y_pred_SVM)
    f1_SVM = f1_score(y_test , y_pred_SVM)
    metric_names = ["accuracy" , "precision" , "recall" ,"f1-score"]
    metrics =[accuracy_SVM , precision_SVM , recall_SVM , f1_SVM]
    colors = ['red', 'blue' , 'green' , 'violet']
    fig , ax = plt.subplots(2,2 , figsize=(12,12))
    ax = ax.flatten()
    fig.suptitle("Metrics for SVM RBF", fontsize=16, fontweight='bold')
    for i in range(len(metric_names)):
        ax[i].bar( metric_names[i] , metrics[i] ,color = colors[i])
        ax[i].set_title(metric_names[i], fontsize=14)

    plt.tight_layout()
    plt.show()
    print("SVM RBF performance metrics:")
    for name, metric in zip(metric_names, metrics):
        print(f"{name}: {metric:.4f}")
plot_confusion_metrics_SVM()


#By obserbving the metrics for the SVM model(which are perfect?) now want to check if there is any overfitting or data leakeage
#So i decided to extract the cross-validation scores using StratifiedKfold with n_splits = 10
def cross_val_svm():
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    cv_scores = cross_val_score(pipe2_rbf, X_train_scaled, y_train, 
                               cv=cv, scoring='accuracy')
    
    print("cross-validation diagnosis SVM:")
    print(f"CV SVM scores: {cv_scores}")
    print(f"CV SVM scores mean: {cv_scores.mean()}")
cross_val_svm()

#This model has almost perfect cv_score_mean for all of the 10 iterations of shuffled data, this means it generilizes well on unseen/shuffled data
#The almost perfect metrics might mean that the dataset may bee too small for the model to learn beacuse this classifier is used for bigger datasets

    
    
