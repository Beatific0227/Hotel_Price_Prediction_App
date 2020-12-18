"""
Name:       Zhengyang Wang
CS230:      Section SN5
Data:       Airbnb Data
URL:        https://share.streamlit.io/beatific0227/final/main

Description:

This program first create a interactive 2D model, then it present with
a interactive 3D model. All column and dots can shows how much price, and
the owner of the house for renting.
Then we have the interactive pivot table where user can drag in any
column that they want and sum, count, average etc...
Lastly, we have a interactive linear regression prediction for the price
base on attitude and longitude, reviews that the house receive
minimum data, and availability. The last price will be printed at the end
so the user can compare whether they are paying extra or less
compare to the price they found on some other renting website.
"""
st.write("This Program is made by Zhengyang Wang, Bentley University, Class of 2022)

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import csv
from pivottablejs import pivot_ui
import streamlit.components.v1 as components
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

MAPKEY = "pk.eyJ1IjoiY2hlY2ttYXJrIiwiYSI6ImNrOTI0NzU3YTA0azYzZ21rZHRtM2tuYTcifQ.6aQ9nlBpGbomhySWPF98DApk.eyJ1IjoiY2hlY2ttYXJrIiwiYSI6ImNrOTI0NzU3YTA0azYzZ21rZHRtM2tuYTcifQ.6aQ9nlBpGbomhySWPF98DA"
FNAME = 'airbnb_cambridge_listings_20201123.csv'
location = []
locations = []

def readData(FNAME):
    with open(FNAME,mode = 'r',encoding='UTF-8') as csv_file:
        data = csv.DictReader(csv_file)
        for row in data:
            location = [(row['name'],float(row['price']),
                        float(row['latitude']),float(row['longitude']))]
            locations.extend(location)
    print(locations)
    return locations

def makemap(locations):
    df = pd.DataFrame(locations, columns = ["Name","Price","lat","lon"])
    print(df)
    st.title("House for Renting on Airbnb")
    st.dataframe(df)
    st.write("Customized Map with Tool Tips and Different Marker Sizes")
    view_state = pdk.ViewState(
        latitude=df["lat"].mean(),
        longitude=df["lon"].mean(),
        zoom=11,
        pitch=50)

    df["scaled_radius"] = df["Price"]/df["Price"].max() * 50
    st.write(df)

    layer1 = pdk.Layer('ScatterplotLayer',
                       data=df,
                       get_position='[lon, lat]',
                       get_radius= 'scaled_radius',
                       radius_scale = 2,
                       radius_min_pixels= 10,
                       radius_max_pixels = 400,
                       get_color=[204,204,255],
                       pickable=True
                         )
    tool_tip = {"html": "Name is {Name} </br/> Price is {Price}</b>",

                "style": { "backgroundColor": "Salmon",
                            "color": "white"}
            }
    map1 = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        mapbox_key=MAPKEY,
        layers= [layer1],
        tooltip= tool_tip
    )
    st.pydeck_chart(map1)

    # Trying to make it 3D
    st.write("Trying Hexagon")
    layer1 = pdk.Layer(
            'ColumnLayer',
            data=df,
            get_position='[lon, lat]',
            get_elevation = 'Price',
            radius = 30,
            elevation_scale=1,
            elevation_range=[0, 1000],
            get_fill_color = [204,204,255],
            pickable=True,
            extruded=True,
         )
    tool_tip = {"html": "Name is {Name} </br/> Price is {Price}</b>",

                "style": { "backgroundColor": "Salmon",
                        "color": "white"}
            }

    map2 = pdk.Deck(
        map_style='mapbox://styles/mapbox/satellite-v9',
        initial_view_state=view_state,
        mapbox_key=MAPKEY,
        layers= layer1,
        tooltip= tool_tip

    )
    st.pydeck_chart(map2)

def InteractivePivotTable(FNAME):
    st.write("Trying an interactive pivot table")
    file = pd.read_csv(FNAME, encoding = 'latin1')
    file.drop(['latitude','longitude','calculated_host_listings_count','id','neighbourhood_group'],inplace=True,axis=1)
    t = pivot_ui(file)
    with open(t.src) as t:
        components.html(t.read(), width=900, height=1000, scrolling=True)
    return t

def RegressionAnalysis(FNAME):
    df = pd.read_csv(FNAME, encoding = 'latin1')
    st.write("Regression Analysis for the price and other variables")
    fig,ax = plt.subplots()
    sns.heatmap(df.corr(),annot=True)
    st.pyplot(fig)
    st.text("Heatmap to visualize the correlation between different attributes"
            "\nwith one the biggest correlation")
    # Point of doing this: to see which variable has a strong correlation with price so I can use them as predictor variable
    # Split dataset into training set and test set:
    # in regression analysis, we use many ways to enhance our model, including backward selection, forward selection, interaction, both way selection
    # test and train is just another way to enhance our model
    X_train, X_test, y_train, y_test = train_test_split(df[['minimum_nights', 'number_of_reviews','availability_365','latitude','longitude']], df['price'], test_size=0.3, random_state=109)
    # Creating the model
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    # Saving the Model
    pickle_out = open("lm.pkl", "wb")
    pickle.dump(lm, pickle_out)
    pickle_out.close()
    pickle_in = open('lm.pkl', 'rb')
    st.sidebar.header('Price Prediction')
    st.title('Price Prediction base on ,minimum nights, number of reviews, amount of host listing and availability')
    st.text("Point of doing this: to see which variable has a strong correlation with price so I can use them as predictor variableSplit dataset into training set\n "
            "and test set:in regression analysis, we use many ways to enhance our model, including backward selection, forward selection, interaction, \n"
            "both way selection test and train is just another way to enhance our model ")

    name = st.text_input("Your Name:")
    minimum_nights = st.number_input("Minimum amount of nights that needs to stay")
    number_of_reviews = st.number_input("number of reviews that the house receive in the past")
    availability_365 =st.number_input("availability for the year (maximum 365):")
    latitude = st.number_input("latitude, start with 42 and . and 4 digits for house locate in cambridge (e.g. 42.3832): ")
    longitude = st.number_input("longitude, start with -71 and . and 4 digits for house locate in cambridge (e.g. -71.1362): ")
    submit = st.button('Predict')

    if submit:
            prediction = lm.predict([[minimum_nights,number_of_reviews,availability_365,latitude,longitude]])
            str(prediction)
            st.write(f"Hello {name}!, your expect price to pay is {prediction}")

def main():
    readData(FNAME)
    makemap(locations)
    InteractivePivotTable(FNAME)
    RegressionAnalysis(FNAME)
main()

st.write("Reference: https://towardsdatascience.com/diabetes-prediction-application-using-streamlit-fed6120124a5\n"
         "\n"
         "https://discuss.streamlit.io/t/is-there-a-way-to-incorporate-pivottablejs-in-streamlit/4461/4")
st.write("This Program is made by Zhengyang Wang, Bentley University, Class of 2022)
