from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
from scipy import stats
import logging
import datetime
import os.path
from flask import Markup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#%matplotlib inline


    
app = Flask(__name__)
app.config["DEBUG"] = True

if __name__ == '__main__':
    app.run(debug = True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

src = os.path.join(BASE_DIR, 'processed_data_new.csv')
data = pd.read_csv(src)


    
def get_life_expectancy(age):
    #data = pd.read_excel("C:/Users/rutuj/Desktop/BE_Project/processed_data_new.xlsx")
    #data.head()

    X = data.iloc[:,3:4]
    y = data.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train) #training the algorithm
    age=age
    arr = [age]
    pd1 = pd.DataFrame(arr)
    y_pred1 = regressor.predict(pd1)
    return y_pred1


@app.route('/', methods=['POST', 'GET'])
def interact_life_expectancy():
    # select box defaults
    default_age = 'Select Age'
    selected_age = default_age

    # data carriers
    string_to_print = ''

    if request.method == 'POST':
        # clean up age field
        selected_age = request.form["age"]
        if (selected_age == default_age):
            selected_age = int(29)
        else:
            selected_age = selected_age


        # estimate lifespan
        predicted_price = get_life_expectancy(age=int(selected_age))

        if (predicted_price is not None):
            # create output string
            string_to_print = Markup("Predicted House Price is  <font size='+10'>" + str(int(np.ceil(predicted_price))) + "</font>Dollars based on previous available data!")
        else:
            string_to_print = Markup("Error! No data found for selected parameters")
            current_time_left = 1


    return render_template('time.html',string_to_print = string_to_print)