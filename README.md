# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/divyadharshiniddanbarasu/basic-nn-model/assets/119393424/4437597d-dd91-4887-912d-ce877e5bf1bc)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Divyadharshini.A
### Register Number: 212222240027

### To Read CSV file from Google Drive :
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

## To train and test
from sklearn.model_selection import train_test_split

## To scale
from sklearn.preprocessing import MinMaxScaler

## To create a neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### Authenticate User:
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :
worksheet = gc.open('Deep Learning').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])

df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()

x=df[['Input']].values
y=df[['Output']].values

x
y

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.4, random_state =35)

Scaler = MinMaxScaler()
Scaler.fit(x_train)

X_train1 = Scaler.transform(x_train)

#Create the model
ai_brain = Sequential([
    Dense(7,activation='relu'),
    Dense(14,activation='relu'),
    Dense(1)
])

#Compile the model
ai_brain.compile(optimizer = 'rmsprop' , loss = 'mse')

# Fit the model
ai_brain.fit(X_train1 , y_train,epochs = 3000)

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()
X_test1 =Scaler.transform(x_test)
ai_brain.evaluate(X_test1,y_test)
X_n1=[[11]]
X_n1_1=Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
## Dataset Information

![image](https://github.com/divyadharshiniddanbarasu/basic-nn-model/assets/119393424/3e33f474-014a-4e92-ab77-5d56bb14b3a6)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/divyadharshiniddanbarasu/basic-nn-model/assets/119393424/007d2121-271e-4a26-bdc4-052a86c9ef1e)


### Test Data Root Mean Squared Error

![image](https://github.com/divyadharshiniddanbarasu/basic-nn-model/assets/119393424/5d069621-e1a3-4caa-a019-d67695de7887)

![image](https://github.com/divyadharshiniddanbarasu/basic-nn-model/assets/119393424/087da099-42c1-45b8-9f0d-df6471cdd9b4)



### New Sample Data Prediction

![image](https://github.com/divyadharshiniddanbarasu/basic-nn-model/assets/119393424/745161d6-7b0d-48f4-a7a5-e8082a1d84d6)


## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
