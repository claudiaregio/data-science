# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Titanic Data Analysis
# ## Goal of Analysis: Use machine learning algorithms to get best accuracy of predictions for who survived the sinking of the Titanic given the attributes in the dataset. 

# %%
#Imports 
import pandas as pd
import numpy as np
import pandas_profiling
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import preprocessing
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy.random import seed

# %% [markdown]
# # Data Analysis

# %%
titanic_df = pd.read_csv("titanic_data.csv")
titanic_df.info()

# %% [markdown]
# ## Data Exploration

# %%
titanic_df.describe()


# %%
# Search for overall trends in the dataset
pandas_profiling.ProfileReport(titanic_df)

# %% [markdown]
# ## Data Cleaning

# %%
#Age is skewed and has a significant number of missing values so best to replace missing values with median of data
age_median = titanic_df['Age'].median(skipna=True)
titanic_df['Age'].fillna(age_median, inplace=True)


# %%
#Cabin has too many missing values and will be completely dropped from the dataframe
titanic_df.drop('Cabin', axis=1, inplace=True)


# %%
#Embarked only has 2 missing values and can be replaced with the most common which is S
titanic_df['Embarked'].fillna("S", inplace=True)


# %%
#Fare has one missing value and can be replaced with the median because it is highly skewed
fare_median = titanic_df['Fare'].median(skipna=True)
titanic_df['Fare'].fillna(fare_median,inplace=True)

# %% [markdown]
# ## Feature Engineering

# %%
#SibSp - Number of siblings/spouses aboard
#Parch - Number of parents/children aboard
#These two variables overlap for every passenger that has this data so I am creating a variable that just detects 
#whether someone is traveling alone or not to account for multicollinearity
titanic_df['TravelGroup']=titanic_df["SibSp"]+titanic_df["Parch"]
titanic_df['TravelAlone']=np.where(titanic_df['TravelGroup']>0, 0, 1) 
titanic_df.head()


# %%
#Does total size of group change the probability of surviving? 
#Initial thought: People who want to check up on the safety of more people take more time looking for them 
#and die as a result of not trying to escape
titanic_df['TravelTotal'] = titanic_df['TravelGroup'] + 1


# %%
#Drop unnecessary variables   - thanks for the help Jeffrey!
titanic_df.drop('SibSp', axis=1, inplace=True)
titanic_df.drop('Parch', axis=1, inplace=True)
titanic_df.drop('TravelGroup', axis=1, inplace=True)
titanic_df.drop('Ticket', axis=1, inplace=True)
titanic_df.drop('Name', axis=1, inplace=True)


# %%
#Hot Encode PClass, Sex, Embarked
le = preprocessing.LabelEncoder()
pclass_cat = le.fit_transform(titanic_df.Pclass)
sex_cat = le.fit_transform(titanic_df.Sex)
embarked_cat = le.fit_transform(titanic_df.Embarked)

#Initialize the encoded categorical columns
titanic_df['pclass_cat'] = pclass_cat
titanic_df['sex_cat'] = sex_cat
titanic_df['embarked_cat'] = embarked_cat

#Drop old categorical fields from dataframe and reindex
dummy_fields = ['Pclass','Sex','Embarked']
data = titanic_df.drop(dummy_fields, axis = 1)
data = titanic_df.reindex(['pclass_cat','sex_cat','Age','Fare','embarked_cat','TravelAlone', 'TravelTotal','Survived'],axis=1)


# %%
data


# %%
#Normalize the continuous variables
continuous = ['Age', 'Fare', 'TravelTotal']

scaler = StandardScaler()

for var in continuous:
    data[var] = data[var].astype('float64')
    data[var] = scaler.fit_transform(data[var].values.reshape(-1, 1))


# %%
data


# %%
#Make sure data is clean/check for null
data[data.isnull().any(axis=1)].head()

# %% [markdown]
# ## Models
# %% [markdown]
# ### Test Train Split

# %%
#Split inputs and output
X = data.iloc[:, 0:7] 
Y = data.iloc[:, 7]


# %%
#Test/Train Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# %% [markdown]
# ### Naive Bayes Classifier

# %%
#NB - All columns
#Initialize + fit model
gnb = GaussianNB().fit(X_train, y_train)

#Predictions
y_pred = gnb.predict(X_test)

#Accuracy Score
NB_all_accuracy = accuracy_score(y_test,y_pred)
print('Naive Bayes Model Accuracy with all attributes: {0:.2f}'.format(NB_all_accuracy))

# %% [markdown]
# ### Decision Tree 

# %%
#DT1 - All attributes
#Initalize + fit model
tree = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 2, random_state=5).fit(X_train, y_train)

#Predictions
y_pred = tree.predict(X_test)

#Accuracy Score
tree_all_accuracy = accuracy_score(y_test, y_pred)
print('Decision Tree Accuracy with all attributes: {0:.2f}'.format(tree_all_accuracy))


# %%
#Tree visualization function
def visualize_tree(tree_data, names):
    dot_data = StringIO()
    export_graphviz(tree_data,out_file=dot_data,
                         feature_names=names,
                         filled=True,rounded=True, 
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())


# %%
#DT1 Graph
names = ['pclass_cat','sex_cat','Age','Fare','embarked_cat','TravelAlone','TravelTotal']
visualize_tree(tree,names)


# %%
print(tree.feature_importances_)


# %%
high_importance = ['sex_cat','Age','Fare']
X_train2 = X_train[high_importance]
X_test2 = X_test[high_importance]


# %%
#DT2 - Top 3 features only
#Initialize + fit model
tree2 = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 2, random_state=5).fit(X_train2, y_train)

#Predictions 
y_pred2 = tree2.predict(X_test2)

#Accuracy Score
tree_imp_accuracy = accuracy_score(y_test, y_pred2)
print('Decision Tree Accuracy with high importance attributes: {0:.2f}'.format(tree_imp_accuracy))


# %%
#DT2 Graph
visualize_tree(tree2,high_importance)

# %% [markdown]
# ### Random Forest

# %%
#RF1 - All attributes
#Initalize + fit model
clf = RandomForestClassifier(n_jobs=2, random_state=0).fit(X_train, y_train)

#Predictions
y_pred = clf.predict(X_test)

#Accuracy Score
RF_all_accuracy = accuracy_score(y_test,y_pred)
print('Random Forest Accuracy with all attributes: {0:.2f}'.format(RF_all_accuracy))


# %%
print(clf.feature_importances_)


# %%
#RF2 - Top 3 features only
#Initialize + fit model
clf2 = RandomForestClassifier(n_jobs=2, random_state=0).fit(X_train2, y_train)

#Predictions
y_pred2 = clf2.predict(X_test2)

#Accuracy Score
RF_imp_accuracy = accuracy_score(y_test,y_pred2)
print('Random Forest Accuracy with high importance attributes: {0:.2f}'.format(RF_imp_accuracy))

# %% [markdown]
# ### Neural Network

# %%
def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):
    # set random seed for reproducibility
    seed(42)

    model = Sequential()
    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
    # create additional hidden layers
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))
    # add dropout, default is none
    model.add(Dropout(dr))
    # create output layer
    model.add(Dense(1, activation='sigmoid'))  # output layer
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# %%
#Initialize and create model
model = create_model()
print(model.summary())


# %%
#Train neural
nn = model.fit(X_train, y_train, epochs=100, validation_split = 0.2, batch_size=32, verbose=0)
nn_accuracy = np.mean(nn.history['val_accuracy'])


# %%
#Summarize history of accuracy
plt.plot(nn.history['accuracy'])
plt.plot(nn.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %% [markdown]
# ## Compare Accuracy Scores
# 

# %%
#All Accuracies
print('NB accuracy: {0:.2f}'.format(NB_all_accuracy))

print("Decision Tree:")
print('All attributes: {0:.2f}'.format(tree_all_accuracy))
print('High importance attributes: {0:.2f}'.format(tree_imp_accuracy))

print("Random Forest:")
print('All attributes: {0:.2f}'.format(RF_all_accuracy))
print('High importance attributes: {0:.2f}'.format(RF_imp_accuracy))

print("Neural Network: ")
print('All attributes: {0:.2f}'.format(nn_accuracy))


