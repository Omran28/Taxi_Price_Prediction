import time
import numpy
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as IDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import pickle

# Load Taxi and Weather data
taxiData = pd.read_csv('taxi-tas-classification-test.csv')
weatherData = pd.read_csv('weather.csv')

# Convert to list to use timestamp
# Time Stamp Null Values
taxiData['time_stamp'].replace(to_replace='', value=taxiData['time_stamp'].mode()[0], inplace=True)
taxiData['time_stamp'].fillna(taxiData['time_stamp'].mode()[0], inplace=True)
taxi = taxiData["time_stamp"].tolist()

weather = weatherData["time_stamp"].tolist()

year = []
month = []
day = []
hour = []
minute = []
second = []

# Convert timestamp in weather to date components
for i in weather:
    dt_object = datetime.fromtimestamp(i)
    year.append(dt_object.strftime("%Y"))
    month.append(dt_object.strftime("%m"))
    day.append(dt_object.strftime("%d"))
    hour.append(dt_object.strftime("%H"))
    minute.append(dt_object.strftime("%M"))
    second.append(dt_object.strftime("%S"))

# Add the components to Weather table
weatherData["year"], weatherData["month"], weatherData["day"], weatherData["hour"], \
weatherData["minute"], weatherData["second"] = [year, month, day, hour, minute, second]

# Take null values of rain
nullValues = weatherData[weatherData["rain"].isna()]

# Take the rest of weather
restValues = weatherData.dropna(how="any", inplace=False)

X1 = restValues[["temp", "wind", "humidity", "clouds"]]
X2 = nullValues[["temp", "wind", "humidity", "clouds"]]
Yrain = restValues['rain']

# Predict rain values
poly_features = PolynomialFeatures(degree=2)
rain_model = pickle.load(open('rain_model.pkl', 'rb'))

X2 = poly_features.fit_transform(X2)
prediction = rain_model.predict(X2)

# Convert negative values
prediction[prediction < 0] = 0.0001

nullValues["rain"] = prediction
frames = [restValues, nullValues]
weatherData = pd.concat(frames)

# taxi-data
year = []
month = []
day = []
hour = []
minute = []
second = []

# Change timestamp in taxi to second format
res = []
for i in taxi:
    res.append(int(i / 1000))

# Convert timestamp in taxi to date components
for t in res:
    obj = datetime.fromtimestamp(t)
    year.append(obj.strftime("%Y"))
    month.append(obj.strftime("%m"))
    day.append(obj.strftime("%d"))
    hour.append(obj.strftime("%H"))
    minute.append(obj.strftime("%M"))
    second.append(obj.strftime("%S"))

# Add the components to Taxi table
taxiData["year"], taxiData["month"], taxiData["day"], taxiData["hour"], \
taxiData["minute"], taxiData["second"] = [year, month, day, hour, minute, second]

taxiData.rename(columns={'source': 'location'}, inplace=True)

# Location null values
locationValues = taxiData['location'].unique()
taxiData['location'].replace(to_replace='', value=taxiData['location'].mode()[0], inplace=True)
taxiData.loc[taxiData["location"].isin(locationValues) == False, ['location']] = taxiData['location'].mode()[0]

# Merge
mergedValues = pd.merge(taxiData, weatherData, on=["location", "month", "day", "hour"])
# Remove duplicated values
Values = mergedValues.drop_duplicates(subset=["destination", "location", "month", "day", "hour"],
                                      keep='first')  # 7877 rows

# Null values preprocessing
# Distance
mean = Values['distance'].replace(to_replace='', value=0)
mean = mean.mean()
Values['distance'].replace(to_replace='', value=mean, inplace=True)
Values = Values.fillna(value=Values['distance'].mean())

# Surge multiplier
mean = Values['surge_multiplier'].replace(to_replace='', value=0)
mean = mean.mean()
Values['surge_multiplier'].replace(to_replace='', value=mean, inplace=True)
Values = Values.fillna(value=Values['surge_multiplier'].mean())

# Cab type
Values['cab_type'].replace(to_replace='', value='Other', inplace=True)
Values.fillna({'cab_type': 'Other'}, inplace=True)

# Name
Values['name'].replace(to_replace='', value=0, inplace=True)
Values.fillna({'name': 0}, inplace=True)

# Converting name (Sum of price)
names = []
for i in range(0, len(Values)):
    d = Values.iloc[i]
    if d.loc['name'] == 'Taxi':
        names.append(2)
    elif d.loc['name'] == 'Shared':
        names.append(3)
    elif d.loc['name'] == 'Black':
        names.append(11)
    elif d.loc['name'] == 'Lux Black XL':
        names.append(13)
    elif d.loc['name'] == 'Lyft XL':
        names.append(8)
    elif d.loc['name'] == 'UberXL':
        names.append(9)
    elif d.loc['name'] == 'Lux':
        names.append(10)
    elif d.loc['name'] == 'Lux Black':
        names.append(12)
    elif d.loc['name'] == 'Black SUV':
        names.append(14)
    elif d.loc['name'] == 'UberX':
        names.append(7)
    elif d.loc['name'] == 'WAV':
        names.append(6)
    elif d.loc['name'] == 'UberPool':
        names.append(4)
    elif d.loc['name'] == 'Lyft':
        names.append(5)
    elif d.loc['name'] == 0:
        names.append(0)
    else:
        names.append(1)

Values["name"] = names

# Converting name (Sum of price)
locations = []
for i in range(0, len(Values)):
    d = Values.iloc[i]
    if d.loc['location'] == 'West End':
        locations.append(3.5)
    elif d.loc['location'] == 'Boston University':
        locations.append(1)
    elif d.loc['location'] == 'Back Bay':
        locations.append(3)
    elif d.loc['location'] == 'Financial District':
        locations.append(2)
    elif d.loc['location'] == 'Northeastern University':
        locations.append(2.5)
    elif d.loc['location'] == 'North End':
        locations.append(6)
    elif d.loc['location'] == 'North Station':
        locations.append(4)
    elif d.loc['location'] == 'Beacon Hill':
        locations.append(5)
    elif d.loc['location'] == 'Theatre District':
        locations.append(3.5)
    elif d.loc['location'] == 'Fenway':
        locations.append(1.5)
    elif d.loc['location'] == 'Haymarket Square':
        locations.append(6.5)
    elif d.loc['location'] == 'South Station':
        locations.append(5.5)
    else:
        locations.append(0.5)

Values["location"] = locations

# Cab type One hot encoding
Values.loc[(Values["cab_type"] != "Uber") & (Values["cab_type"] != "Lyft"), "cab_type"] = "Other"

# New values of cab type

var = 1
if len(Values["cab_type"].unique()) > 2:
    var = 2
#
label = LabelEncoder()
int_data = label.fit_transform(Values['cab_type'])
int_data = int_data.reshape(len(int_data), 1)
one_hot_data = OneHotEncoder(sparse=False)
one_hot_data = one_hot_data.fit_transform(int_data)
pickle.dump(one_hot_data, open("encoder.pkl", "wb"))

Values["uber"] = one_hot_data[:, var]
Values["lyft"] = one_hot_data[:, 0]

# Drop unnecessary columns
Values = Values.drop(
    ["cab_type", "destination", "id", "product_id", "time_stamp_x", "time_stamp_y", "year_x", "year_y", "minute_x",
     "minute_y", "second_x", "second_y"], axis=1)
x = False
val = Values["name"].unique()
for i in val:
    if i == "Taxi":
        x = True

if x:
    # Taxi Prediction
    options = ['cheap', 'moderate', 'expensive', 'very expensive']
    unknown_category = Values[Values["RideCategory"].isin(options) == False]
    known_category = Values[Values["RideCategory"].isin(options)]

    classifier = pickle.load(open('unknown_prediciton.pkl', 'rb'))

    X = unknown_category.iloc[:, Values.columns != 'RideCategory']
    poly_features = PolynomialFeatures(degree=2)
    X = poly_features.fit_transform(X)

    prediction = classifier.predict(X)
    unknown_category["RideCategory"] = prediction

    frames = [unknown_category, known_category]
    Values = pd.concat(frames)

# Modeling
Y = []
X = Values.iloc[:, Values.columns != 'RideCategory']
for i in Values["RideCategory"]:
    Y.append(i)

# Feature Incremental
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)

# Classification Selection (Select From Model (L1-based feature selection))
model = pickle.load(open('classification_selection.pkl', 'rb'))
X_poly = model.transform(X_poly)

Values = pd.DataFrame(numpy.array(X_poly))

Values["RideCategory"] = Y
X = Values.loc[:, Values.columns != 'RideCategory']

# Separate data into test and train splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

# Reset lists
Model_Name = []
Classification_Accuracy = []
Time_Training = []
Time_Testing = []

# Models
# Poly SVC
t = time.time()
svc = pickle.load(open('SVC Poly_3_0.9.pkl', 'rb'))
Time_Training.append(time.time() - t)

t = time.time()
testingPrediction = svc.predict(X_test)
Time_Testing.append(time.time() - t)

Accuracy = accuracy_score(testingPrediction, Y_test)
Model_Name.append("SVC Poly")
Classification_Accuracy.append(Accuracy)
print("SVC poly testing accuracy: ", Accuracy)

trainingPrediction = svc.predict(X_train)
print("SVC poly training accuracy: ", accuracy_score(trainingPrediction, Y_train))

# DTC
t = time.time()
classifier = pickle.load(open('DTC_7_7.pkl', 'rb'))
Time_Training.append(time.time() - t)

t = time.time()
y_pred = classifier.predict(X_test)
Time_Testing.append(time.time() - t)

Accuracy = accuracy_score(y_pred, Y_test)
Model_Name.append("DTC")

Classification_Accuracy.append(Accuracy)
print("DTC testing accuracy: ", Accuracy)

y_pred = classifier.predict(X_train)
print("DTC training accuracy: ", accuracy_score(y_pred, Y_train))

# KNN
t = time.time()
classifier = pickle.load(open('KNN_40_5.pkl', 'rb'))
Time_Training.append(time.time() - t)

t = time.time()
y_pred = classifier.predict(X_test)
Time_Testing.append(time.time() - t)

Accuracy = accuracy_score(y_pred, Y_test)
Model_Name.append("KNN")
Classification_Accuracy.append(Accuracy)
print("K Neighbours testing accuracy: ", Accuracy)

y_pred = classifier.predict(X_train)
print("K Neighbours training accuracy: ", accuracy_score(y_pred, Y_train))

# Linear Discriminant Analysis
IDA = IDA(n_components=1)
X_train = IDA.fit_transform(X_train, Y_train)
X_test = IDA.transform(X_test)

# Model Linear Discriminant Analysis
t = time.time()
classifier = pickle.load(open('IDA_2_0.pkl', 'rb'))
Time_Training.append(time.time() - t)

t = time.time()
y_pred = classifier.predict(X_test)
Time_Testing.append(time.time() - t)

Accuracy = accuracy_score(y_pred, Y_test)
Model_Name.append("IDA")
Classification_Accuracy.append(Accuracy)
print("LDA testing accuracy: ", Accuracy)

y_pred = classifier.predict(X_train)
print("LDA training accuracy: ", accuracy_score(y_pred, Y_train))

# Classification Accuracy Graph
fig1 = plt.figure(figsize=(10, 5))
plt.bar(Model_Name, Classification_Accuracy, color='maroon', width=0.4)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.show()

# Test Time Graph
fig2 = plt.figure(figsize=(10, 5))
plt.bar(Model_Name, Time_Testing, color='maroon', width=0.4)
plt.xlabel("Models")
plt.ylabel("Test Time")
plt.title("Test Time")
plt.show()

# Training Time Graph
fig3 = plt.figure(figsize=(10, 5))
plt.bar(Model_Name, Time_Training, color='maroon', width=0.4)
plt.xlabel("Models")
plt.ylabel("Training Time")
plt.title("Training Time")
plt.show()

# Reset lists
Time_Testing.clear()
Time_Training.clear()
Classification_Accuracy.clear()
Model_Name.clear()

# # Decision Tree Classifier (HyperParameters)
# # 1 DTC
# t = time.time()
# classifier = pickle.load(open('DTC_5_10.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Model_Name.append("DTC_5_10")
# Accuracy = accuracy_score(y_pred, Y_test)
# Classification_Accuracy.append(Accuracy)
# print("DTC_5_10 testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("DTC_5_10 training accuracy: ", accuracy_score(y_pred, Y_train))
# print("***********************************************************************************")
#
# # 2 DTC
# t = time.time()
# classifier = pickle.load(open('DTC_5_6.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("DTC_5_6")
# Classification_Accuracy.append(Accuracy)
# print("DTC_5_6 testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("DTC_5_6 training accuracy: ", accuracy_score(y_pred, Y_train))
# print("***********************************************************************************")
#
# # 3 DTC
# t = time.time()
# classifier = pickle.load(open('DTC_5_2.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("DTC_5_2")
# Classification_Accuracy.append(Accuracy)
# print("DTC_5_2 testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("DTC_5_2 training accuracy: ", accuracy_score(y_pred, Y_train))
# print("***********************************************************************************")
#
# # 4 DTC
# t = time.time()
# classifier = pickle.load(open('DTC_14_7.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("DTC_14_7")
#
# Classification_Accuracy.append(Accuracy)
# print("DTC_14_7 testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("DTC_14_7 training accuracy: ", accuracy_score(y_pred, Y_train))
# print("***********************************************************************************")
#
# # 5 DTC
# t = time.time()
# classifier = pickle.load(open('DTC_7_7.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("DTC_7_7")
#
# Classification_Accuracy.append(Accuracy)
# print("DTC_7_7 testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("DTC_7_7 training accuracy: ", accuracy_score(y_pred, Y_train))
# print("***********************************************************************************")
#
# # 6 DTC
# t = time.time()
# classifier = pickle.load(open('DTC_1_7.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("DTC_1_7")
# Classification_Accuracy.append(Accuracy)
# print("DTC_1_7 testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("DTC_1_7 training accuracy: ", accuracy_score(y_pred, Y_train))
# print("-----------------------------------------------------------------------------")
#
# # Classification Accuracy Graph DTC
# fig1 = plt.figure(figsize=(10, 5))
# plt.bar(Model_Name, Classification_Accuracy, color='maroon', width=0.4)
# plt.xlabel("Models")
# plt.ylabel("Accuracy")
# plt.title("DTC")
# plt.show()
#
# # Test Time Graph DTC
# fig2 = plt.figure(figsize=(10, 5))
# plt.bar(Model_Name, Time_Testing, color='maroon', width=0.4)
# plt.xlabel("Models")
# plt.ylabel("Test Time")
# plt.title("DTC")
# plt.show()
#
# # Training Time Graph DTC
# fig3 = plt.figure(figsize=(10, 5))
# plt.bar(Model_Name, Time_Training, color='maroon', width=0.4)
# plt.xlabel("Models")
# plt.ylabel("Training Time")
# plt.title("DTC")
# plt.show()
#
# # Reset lists
# Time_Testing.clear()
# Time_Training.clear()
# Classification_Accuracy.clear()
# Model_Name.clear()
#
# # Poly SVC
# # 1 Poly SVC
# t = time.time()
# svc = pickle.load(open('SVC Poly_1_0.5.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# testingPrediction = svc.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(testingPrediction, Y_test)
# Model_Name.append("SVC Poly_1_0.5")
# Classification_Accuracy.append(Accuracy)
# print("SVC poly testing accuracy: ", Accuracy)
#
# trainingPrediction = svc.predict(X_train)
# print("SVC poly training accuracy: ", accuracy_score(trainingPrediction, Y_train))
# print("***********************************************************************************")
#
# # 2 Poly SVC
# t = time.time()
# svc = pickle.load(open('SVC Poly_10_0.5.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# testingPrediction = svc.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(testingPrediction, Y_test)
# Model_Name.append("SVC Poly_10_0.5")
# Classification_Accuracy.append(Accuracy)
# print("SVC poly testing accuracy: ", Accuracy)
#
# trainingPrediction = svc.predict(X_train)
# print("SVC poly training accuracy: ", accuracy_score(trainingPrediction, Y_train))
# print("***********************************************************************************")
#
# # 3 Poly SVC
# t = time.time()
# svc = pickle.load(open('SVC Poly_20_0.5.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# testingPrediction = svc.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(testingPrediction, Y_test)
# Model_Name.append("SVC Poly_20_0.5")
# Classification_Accuracy.append(Accuracy)
# print("SVC poly testing accuracy: ", Accuracy)
#
# trainingPrediction = svc.predict(X_train)
# print("SVC poly training accuracy: ", accuracy_score(trainingPrediction, Y_train))
# print("***********************************************************************************")
#
# # 4 Poly SVC
# t = time.time()
# svc = pickle.load(open('SVC Poly_3_0.1.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# testingPrediction = svc.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(testingPrediction, Y_test)
# Model_Name.append("SVC Poly_3_0.1")
# Classification_Accuracy.append(Accuracy)
# print("SVC poly testing accuracy: ", Accuracy)
#
# trainingPrediction = svc.predict(X_train)
# print("SVC poly training accuracy: ", accuracy_score(trainingPrediction, Y_train))
# print("***********************************************************************************")
#
# # 5 Poly SVC
# t = time.time()
# svc = pickle.load(open('SVC Poly_3_0.6.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# testingPrediction = svc.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(testingPrediction, Y_test)
# Model_Name.append("SVC Poly_3_0.6")
# Classification_Accuracy.append(Accuracy)
# print("SVC poly testing accuracy: ", Accuracy)
#
# trainingPrediction = svc.predict(X_train)
# print("SVC poly training accuracy: ", accuracy_score(trainingPrediction, Y_train))
# print("***********************************************************************************")
#
# # 6 Poly SVC
# t = time.time()
# svc = pickle.load(open('SVC Poly_3_0.9.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# testingPrediction = svc.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(testingPrediction, Y_test)
# Model_Name.append("SVC Poly_3_0.9")
# Classification_Accuracy.append(Accuracy)
# print("SVC poly testing accuracy: ", Accuracy)
#
# trainingPrediction = svc.predict(X_train)
# print("SVC poly training accuracy: ", accuracy_score(trainingPrediction, Y_train))
# print("-----------------------------------------------------------------------------")
#
# # Classification Accuracy Graph for Poly SVC
# fig1 = plt.figure(figsize=(10, 5))
# plt.bar(Model_Name, Classification_Accuracy, color='maroon', width=0.4)
# plt.xlabel("Models")
# plt.ylabel("Accuracy")
# plt.title("Poly SVC")
# plt.show()
#
# # Test Time Graph
# fig2 = plt.figure(figsize=(10, 5))
# plt.bar(Model_Name, Time_Testing, color='maroon', width=0.4)
# plt.xlabel("Models")
# plt.ylabel("Test Time")
# plt.title("Poly SVC")
# plt.show()
#
# # Training Time Graph
# fig3 = plt.figure(figsize=(10, 5))
# plt.bar(Model_Name, Time_Training, color='maroon', width=0.4)
# plt.xlabel("Models")
# plt.ylabel("Training Time")
# plt.title("Poly SVC")
# plt.show()
#
# # Reset lists
# Time_Testing.clear()
# Time_Training.clear()
# Classification_Accuracy.clear()
# Model_Name.clear()
#
#
# # Model K-Nearest Neighbors (KNN)
# # 1(KNN)
# t = time.time()
# classifier = pickle.load(open('KNN_29_1.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("KNN_29_1")
# Classification_Accuracy.append(Accuracy)
# print("K Neighbours testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("K Neighbours training accuracy: ", accuracy_score(y_pred, Y_train))
# print("***********************************************************************************")
#
# # 2 (KNN)
# t = time.time()
# classifier = pickle.load(open('KNN_29_5.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("KNN_29_5")
# Classification_Accuracy.append(Accuracy)
# print("K Neighbours testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("K Neighbours training accuracy: ", accuracy_score(y_pred, Y_train))
# print("***********************************************************************************")
#
# # 3 (KNN)
# t = time.time()
# classifier = pickle.load(open('KNN_29_12.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("KNN_29_12")
# Classification_Accuracy.append(Accuracy)
# print("K Neighbours testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("K Neighbours training accuracy: ", accuracy_score(y_pred, Y_train))
# print("***********************************************************************************")
#
# # 4 (KNN)
# t = time.time()
# classifier = pickle.load(open('KNN_5_5.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("KNN_5_5")
# Classification_Accuracy.append(Accuracy)
# print("K Neighbours testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("K Neighbours training accuracy: ", accuracy_score(y_pred, Y_train))
# print("***********************************************************************************")
#
# # 5(KNN)
# t = time.time()
# classifier = pickle.load(open('KNN_20_5.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("KNN_20_5")
# Classification_Accuracy.append(Accuracy)
# print("K Neighbours testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("K Neighbours training accuracy: ", accuracy_score(y_pred, Y_train))
# print("***********************************************************************************")
#
# # 6 (KNN)
# t = time.time()
# classifier = pickle.load(open('KNN_40_5.pkl', 'rb'))
# Time_Training.append(time.time() - t)
#
# t = time.time()
# y_pred = classifier.predict(X_test)
# Time_Testing.append(time.time() - t)
#
# Accuracy = accuracy_score(y_pred, Y_test)
# Model_Name.append("KNN_40_5")
# Classification_Accuracy.append(Accuracy)
# print("K Neighbours testing accuracy: ", Accuracy)
#
# y_pred = classifier.predict(X_train)
# print("K Neighbours training accuracy: ", accuracy_score(y_pred, Y_train))
# print("-----------------------------------------------------------------------------")
#
# # Classification Accuracy Graph for KNN
# fig1 = plt.figure(figsize=(10, 5))
# plt.bar(Model_Name, Classification_Accuracy, color='maroon', width=0.4)
# plt.xlabel("Models")
# plt.ylabel("Accuracy")
# plt.title("KNN")
# plt.show()
#
# # Test Time Graph
# fig2 = plt.figure(figsize=(10, 5))
# plt.bar(Model_Name, Time_Testing, color='maroon', width=0.4)
# plt.xlabel("Models")
# plt.ylabel("Test Time")
# plt.title("KNN")
# plt.show()
#
# # Training Time Graph
# fig3 = plt.figure(figsize=(10, 5))
# plt.bar(Model_Name, Time_Training, color='maroon', width=0.4)
# plt.xlabel("Models")
# plt.ylabel("Training Time")
# plt.title("KNN")
# plt.show()
