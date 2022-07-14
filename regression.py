from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
import pandas as pd
from sklearn.metrics import r2_score
import joblib

# Load Taxi and Weather data
taxiData = pd.read_csv('taxi-rides.csv')
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
Y = restValues['rain']

# Predict rain values
# Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.20, shuffle=True, random_state=10)
# Polynomial linear regression Model
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
pickle.dump(poly_model, open("rain_model.pkl", "wb"))

X2 = poly_features.fit_transform(X2)
prediction = poly_model.predict(X2)

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

# Change timestamp to second format
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

# Remove duplicated
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

# Cab type One hot encoding
label = LabelEncoder()
int_data = label.fit_transform(Values['cab_type'])
int_data = int_data.reshape(len(int_data), 1)
one_hot_data = OneHotEncoder(sparse=False)
one_hot_data = one_hot_data.fit_transform(int_data)
joblib.dump(one_hot_data, 'regression_encoder.joblib')

# New values of cab type
Values["uber"] = one_hot_data[:, 0]
Values["lyft"] = one_hot_data[:, 1]
Values = Values.drop(
    ["cab_type", "destination", "id", "product_id", "time_stamp_x", "time_stamp_y", "year_x", "year_y", "minute_x",
     "minute_y", "second_x", "second_y"], axis=1)

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

# Feature Selection
X = Values.iloc[:, Values.columns != 'price']
Y = Values.iloc[:, 5]

# Regression Selection
select = VarianceThreshold(threshold=1)
select.fit(X, Y)
pickle.dump(select, open("regression_selection.pkl", "wb"))
topFeatures = select.get_feature_names_out()
Values = Values[topFeatures]

# model data needed
all_X = Values.loc[:, Values.columns != 'price']

# separate data
all_X_train, all_X_test, all_y_train, all_y_test = train_test_split(all_X, Y, test_size=0.30, shuffle=True,
                                                                    random_state=10)
all_poly_features = PolynomialFeatures(degree=3)
all_X_train_poly = all_poly_features.fit_transform(all_X_train)

# linear regression model
poly_model = linear_model.LinearRegression()
poly_model.fit(all_X_train_poly, all_y_train)
pickle.dump(poly_model, open("regression_linear_model.pkl", "wb"))

# BayesianRidge model
BR_model = BayesianRidge(compute_score=True)
BR_model.fit(all_X_train_poly, all_y_train)
pickle.dump(BR_model, open("regression_bayesian_model.pkl", "wb"))


# prediction
lin_model_prediction = poly_model.predict(all_poly_features.fit_transform(all_X_test))
BR_model_prediction = BR_model.predict(all_poly_features.fit_transform(all_X_test))

# Error of each model
print('Mean Square Error of linear_model_prediction', metrics.mean_squared_error(all_y_test, lin_model_prediction))
print('R2 score of linear_model_prediction', r2_score(all_y_test, lin_model_prediction))
print('Mean Square Error of BR_model_prediction', metrics.mean_squared_error(all_y_test, BR_model_prediction))
print('R2 score of BR_model_prediction', r2_score(all_y_test, BR_model_prediction))
