#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# <h1 style="text-align: center;">Deep Learning<br><br>Assignment-1 (ANN)<br><br>Churn Prediction for Bank Customer<br><h1>

# # Dataset Info

# We have a dataset in which there are details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.
# 
# The features in the given dataset are:
# - **rownumber:** Row Numbers from 1 to 10000.
# - **customerid:** A unique ID that identifies each customer.
# - **surname:** The customer’s surname.
# - **creditscore:** A credit score is a number between 300–850 that depicts a consumer's creditworthiness.
# - **geography:** The country from which the customer belongs to.
# - **Gender:** The customer’s gender: Male, Female
# - **Age:** The customer’s current age, in years, at the time of being customer.
# - **tenure:** The number of years for which the customer has been with the bank.
# - **balance:** Bank balance of the customer.
# - **numofproducts:** the number of bank products the customer is utilising.
# - **hascrcard:** The number of credit cards given to the customer by the bank.
# - **isactivemember:** Binary Flag for indicating if the client is active or not with the bank before the moment where the client exits the company (recorded in the variable "exited")
# - **exited:** Binary flag 1 if the customer closed account with bank and 0 if the customer is retained.

# # Improt Libraries & Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the dataset
data = pd.read_csv('Churn_Modelling.csv')
data.head()


# # Exploratory Data Analysis and Visualization

# 1. Implement basic steps to see how is your data looks like
# 2. Check for missing values
# 3. Drop the features that not suitable for modelling
# 4. Implement basic visualization steps such as histogram, countplot, heatmap
# 5. Convert categorical variables to dummy variables

# In[3]:


# Checking the shape of the dataset
shape_info = data.shape

# Checking the data types
data_types = data.dtypes

# Statistical summary
summary = data.describe()

shape_info, data_types, summary


# In[4]:


# Checking for missing values
missing_values = data.isnull().sum()

missing_values


# In[5]:


# Dropping unnecessary columns
data_cleaned = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

data_cleaned.head()


# In[6]:


# Setting up the figure size
plt.figure(figsize=(15, 10))

# Histograms for numerical features
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data_cleaned[feature], bins=30, kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[7]:


# Setting up the figure size
plt.figure(figsize=(15, 10))

# Countplots for categorical and binary features
categorical_binary_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Exited']
for i, feature in enumerate(categorical_binary_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=data_cleaned, x=feature)
    plt.title(f'Countplot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[8]:


# Heatmap for correlations
plt.figure(figsize=(12, 8))
sns.heatmap(data_cleaned.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# In[9]:


# One-hot encoding for categorical variables
data_encoded = pd.get_dummies(data_cleaned, columns=['Geography', 'Gender'], drop_first=True)

data_encoded.head()


# # Preprocessing of Data
# - Train | Test Split, Scalling

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Splitting the data into features (X) and target (y)
X = data_encoded.drop('Exited', axis=1)
y = data_encoded['Exited']

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape, X_test_scaled.shape


# # Modelling & Model Performance

# ## without class_weigth

# ### Create The Model

# In[38]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall
from tensorflow.keras.layers import Dense

# Initialize the neural network model
model = Sequential()

# Input layer and first hidden layer
model.add(Dense(units=6, activation='relu', input_dim=11))

# Second hidden layer
model.add(Dense(units=6, activation='relu'))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',Recall(name='recall')])

model.summary()


# In[39]:


# Train the model
history = model.fit(X_train_scaled, y_train, batch_size=10, epochs=100, validation_data=(X_test_scaled, y_test))


# ### Evaluate The Model
# 
# - Plot the model history to observe the changing of metrics
# - Make prediction to see "confusion matrix" and "classification report"
# - Check ROC (Receiver Operating Curve) and AUC (Area Under Curve) for the model

# In[40]:


# Plotting the training history
plt.figure(figsize=(14, 5))

# Plotting the training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plotting the training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix

# Making predictions
y_pred_probs = model.predict(X_test_scaled)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Classification Report
print(classification_report(y_test, y_pred))


# In[45]:


from sklearn.metrics import roc_curve, roc_auc_score

# Getting the probabilities
y_pred_probs = model.predict(X_test_scaled)

# Computing ROC curve and ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = roc_auc_score(y_test, y_pred_probs)

# Plotting the ROC curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# ## with class_weigth
# 
# Investigate how the "class_weight" hyper-parameter is used in a Neural Network.

# ### Create The Model

# In[46]:


from sklearn.utils.class_weight import compute_class_weight

# Calculating class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}


# In[47]:


# Train the model with class weights
history_weighted = model.fit(X_train_scaled, y_train, batch_size=10, epochs=100, validation_data=(X_test_scaled, y_test), class_weight=class_weights_dict)


# ### Evaluate The Model
# 
# - Plot the model history to observe the changing of metrics
# - Make prediction to see "confusion matrix" and "classification report"
# - Check ROC (Receiver Operating Curve) and AUC (Area Under Curve) for the model

# In[48]:


# Plotting the training history with class weights
plt.figure(figsize=(14, 5))

# Plotting the training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history_weighted.history['accuracy'], label='Train Accuracy')
plt.plot(history_weighted.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy with Class Weights')

# Plotting the training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history_weighted.history['loss'], label='Train Loss')
plt.plot(history_weighted.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss with Class Weights')

plt.show()


# In[49]:


# Making predictions with class weights
y_pred_ = model.predict(X_test_scaled)
y_pred_weighted = (y_pred_ > 0.5).astype(int).flatten()

# Confusion Matrix
print(confusion_matrix(y_test, y_pred_weighted))

# Classification Report
print(classification_report(y_test, y_pred_weighted))


# In[50]:


# Getting the probabilities with class weights
y_pred_probs_weighted = model.predict(X_test_scaled)

# Computing ROC curve and ROC AUC
fpr_weighted, tpr_weighted, thresholds_weighted = roc_curve(y_test, y_pred_probs_weighted)
roc_auc_weighted = roc_auc_score(y_test, y_pred_probs_weighted)

# Plotting the ROC curve
plt.figure(figsize=(10, 7))
plt.plot(fpr_weighted, tpr_weighted, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_weighted:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve with Class Weights')
plt.legend(loc="lower right")
plt.show()


# ## Implementation Different Methods to Develop The Model
# 
# - Implement the following methods on model creating with "class_weight" parameter
# - Create and evaluate model for each method

# ### Increase The Learning Rate and Observe The Results

# In[51]:


from tensorflow.keras.optimizers import Adam

# Initialize the neural network model
model_high_lr = Sequential()

# Layers
model_high_lr.add(Dense(units=6, activation='relu', input_dim=11))
model_high_lr.add(Dense(units=6, activation='relu'))
model_high_lr.add(Dense(units=1, activation='sigmoid'))

# Compile the model with a higher learning rate
optimizer_high_lr = Adam(learning_rate=0.01)
model_high_lr.compile(optimizer=optimizer_high_lr, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with class weights
history_high_lr = model_high_lr.fit(X_train_scaled, y_train, batch_size=10, epochs=100, validation_data=(X_test_scaled, y_test), class_weight=class_weights_dict)


# ### Add Dropout Layer

# In[52]:


from tensorflow.keras.layers import Dropout

# Initialize the model with dropout
model_dropout = Sequential()

# Layers with dropout
model_dropout.add(Dense(units=6, activation='relu', input_dim=11))
model_dropout.add(Dropout(0.2))
model_dropout.add(Dense(units=6, activation='relu'))
model_dropout.add(Dropout(0.2))
model_dropout.add(Dense(units=1, activation='sigmoid'))

# Compile
model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history_dropout = model_dropout.fit(X_train_scaled, y_train, batch_size=10, epochs=100, validation_data=(X_test_scaled, y_test), class_weight=class_weights_dict)


# ### Add Early Stop

# #### Monitor the "val_loss" as metric

# In[44]:


from tensorflow.keras.callbacks import EarlyStopping

early_stop_loss = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

history_early_stop_loss = model.fit(X_train_scaled, y_train, batch_size=10, epochs=500, validation_data=(X_test_scaled, y_test), callbacks=[early_stop_loss], class_weight=class_weights_dict)


# #### Monitor the "val_recall" as metric

# In[43]:


early_stop_recall = EarlyStopping(monitor='val_recall', mode='max', verbose=1, patience=25)

history_early_stop_recall = model.fit(X_train_scaled, y_train, batch_size=10, epochs=500, validation_data=(X_test_scaled, y_test), callbacks=[early_stop_recall], class_weight=class_weights_dict)


# ## Optuna

# In[53]:


import optuna

def objective(trial):
    # Define hyperparameters using trial.suggest_...
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    units_layer1 = trial.suggest_int('units_layer1', 4, 12)
    units_layer2 = trial.suggest_int('units_layer2', 4, 12)
    
    # Create and compile the model
    model_optuna = Sequential()
    model_optuna.add(Dense(units=units_layer1, activation='relu', input_dim=11))
    model_optuna.add(Dense(units=units_layer2, activation='relu'))
    model_optuna.add(Dense(units=1, activation='sigmoid'))
    model_optuna.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model and return a metric to be optimized
    history = model_optuna.fit(X_train_scaled, y_train, batch_size=10, epochs=50, validation_data=(X_test_scaled, y_test), verbose=0, class_weight=class_weights_dict)
    
    # Return the validation accuracy for the last epoch as the metric to be optimized
    return history.history['val_accuracy'][-1]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print out the best hyperparameters
print(study.best_params)


# ### Evaluate The Model
# 
# - Plot the model history to observe the changing of metrics
# - Make prediction to see "confusion matrix" and "classification report"
# - Check ROC (Receiver Operating Curve) and AUC (Area Under Curve) for the model

# In[54]:


best_params = study.best_params

# Train the model with the best parameters
model_best = Sequential()
model_best.add(Dense(units=best_params['units_layer1'], activation='relu', input_dim=11))
model_best.add(Dense(units=best_params['units_layer2'], activation='relu'))
model_best.add(Dense(units=1, activation='sigmoid'))
model_best.compile(optimizer=Adam(learning_rate=best_params['lr']), loss='binary_crossentropy', metrics=['accuracy'])
history_best = model_best.fit(X_train_scaled, y_train, batch_size=10, epochs=100, validation_data=(X_test_scaled, y_test), verbose=1, class_weight=class_weights_dict)

# Plotting the training history
plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_best.history['accuracy'], label='Train Accuracy')
plt.plot(history_best.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_best.history['loss'], label='Train Loss')
plt.plot(history_best.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()


# In[56]:


from sklearn.metrics import classification_report, confusion_matrix

# Making predictions
y_pred_b = model.predict(X_test_scaled)
y_pred_best = (y_pred_ > 0.5).astype(int).flatten()


# Confusion Matrix
print(confusion_matrix(y_test, y_pred_best))

# Classification Report
print(classification_report(y_test, y_pred_best))


# In[57]:


from sklearn.metrics import roc_curve, roc_auc_score

# Getting the probabilities
y_pred_probs_best = model_best.predict(X_test_scaled)

# Compute ROC curve and ROC AUC
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_pred_probs_best)
roc_auc_best = roc_auc_score(y_test, y_pred_probs_best)

# Plot
plt.figure(figsize=(10, 7))
plt.plot(fpr_best, tpr_best, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_best:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___
