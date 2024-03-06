#Pandas library
import pandas as pd
#Numpy Library
import numpy as np
#pickle library to load ML model
import pickle
#Dashboard Libraries
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

#Json library-to get response from OneMapApi- requests(latitude and longitude coord)
import json
import requests

#to calculate min dist
from geopy.distance import geodesic

import statistics

#Reading MRT_coordinates data
data=pd.read_csv("mrt.csv")
mrt_location=pd.DataFrame(data)

# Configuring Streamlit GUI

st.set_page_config(layout="wide")

#Menu options

st.header(":blue[Singapore Resale Flat Prices Prediction]")
selected = option_menu(None,
                           options = ["Home","Prediction","Insights"],
                           icons = ["house","currency-rupee","archive"],
                           default_index=0,
                           orientation="horizontal",
                           styles={"container": {"width": "100%"},
                                   "icon": {"color": "white", "font-size": "24px"},
                                   "nav-link": {"font-size": "15px", "text-align": "center", "margin": "-2px"},
                                   "nav-link-selected": {"background-color": "#6F36AD"}})
# # # MENU 1 - Home
if selected == "Home":
    col1,col2 = st.columns(2)
    with col1:
        st.header("Overview")
        st.write("This project aims to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore")
        st.header("Problem Statement")
        st.write("The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat. There are many factors that can affect resale prices, such as location, flat type, floor area, and lease duration. A predictive model can help to overcome these challenges by providing users with an estimated resale price based on these factors.")
        st.write("Therefore, by using Decision-Tree Regresssion Model, can find out how the selling price of a HDB resale flat changes based on its following characteristics:")
        st.write("1.its distance to the Central Business District (CBD)")
        st.write("2.its distance to the nearest MRT station")
        st.write("3.its flat size")
        st.write("4.its floor level")
        st.write("5.its remaining years of lease")

    with col2:
        st.header("Technologies Used")
        st.write("Python,Pandas,Numpy,Scikit-Learn,Streamlitm,Machine Learning,Visualization,EDA")
        st.header("Machine Learning Model")
        st.write("Decision Tree Regression is used to predict the resale price")
        st.write("The model got better accuracy compared with others")
        

#Menu-2 - Prediction
if selected=="Prediction":
    street=['ANG MO KIO AVE 4', 'ANG MO KIO AVE 10', 'ANG MO KIO AVE 5',
       'ANG MO KIO AVE 8', 'ANG MO KIO AVE 1', 'ANG MO KIO AVE 3',
       'ANG MO KIO AVE 6', 'ANG MO KIO ST 52', 'ANG MO KIO ST 21',
       'ANG MO KIO ST 31', 'BEDOK RESERVOIR RD', 'BEDOK STH RD',
       'BEDOK NTH ST 3', 'BEDOK NTH AVE 1', 'BEDOK NTH RD',
       'NEW UPP CHANGI RD', 'CHAI CHEE ST', 'BEDOK NTH ST 1',
       'BEDOK NTH AVE 4', 'BEDOK NTH ST 2', 'CHAI CHEE AVE',
       'BEDOK NTH AVE 3', 'BEDOK STH AVE 1', 'BEDOK CTRL']
    town=['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
       'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
       'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
       'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    flat=['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
       'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
       'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
       'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
       'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen']
    storey=['07 TO 09', '01 TO 03', '13 TO 15', '10 TO 12', '04 TO 06',
       '19 TO 21', '16 TO 18', '22 TO 24', '25 TO 27', '28 TO 30',
       '34 TO 36', '46 TO 48', '31 TO 33', '37 TO 39', '43 TO 45',
       '40 TO 42', '49 TO 51']
    date=[1986, 1981, 1980, 1979, 1978, 1985, 1976, 1977, 2002, 1993, 1996,
       2006, 2003, 1982, 1974, 2010, 1987, 1984, 2000, 1989, 1995, 1992,
       1988, 1998, 1990, 1983, 2004, 1997, 2005, 1969, 1970, 1971, 1973,
       2009, 1999, 2001, 2008, 2007, 1975, 2011, 1968, 1967, 1972, 1991,
       2012, 1994, 1966, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022,
       2020]
    try:
        with st.form("form1"):
            col1,col2=st.columns(2)
        with col1:
            # -----New Data inputs from the user for predicting the resale price-----
            #street_name = st.text_input("Street Name")
            street_name=st.selectbox("Street_Name", sorted(street),key=2)
            town=st.selectbox("Town_Name", sorted(town),key=3)
            flat=st.selectbox("Flat_Model", sorted(flat),key=4)
            floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
        with col2:
            storey_range=st.selectbox("Storey_Range", sorted(storey),key=5)
            lease_commence_date=st.selectbox("Lease_Commence_Date", sorted(date),key=6)
            block = st.text_input("Block Number")
            #lease_commence_date = st.number_input('Lease Commence Date')
            #storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

            # -----Submit Button for PREDICT RESALE PRICE-----
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")
            
            

            if submit_button:
                with open("model.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open("scaler.pkl", 'rb') as f:
                    scaler_loaded = pickle.load(f)
                    
                
#calculate remaining lease years:
            lease_remain_years = 99 - (2024 - lease_commence_date)
            
#calculate median value for storey_Range
            split_list = storey_range.split(' TO ')
            float_list = [float(i) for i in split_list]
            storey_median = statistics.median(float_list)
            
#Getting the address by joining the block number and the street name
            address=block + " " + street_name
            query_address=address
            query_string = 'https://www.onemap.gov.sg/api/common/elastic/search?searchVal='+str(query_address)+'&returnGeom=Y&getAddrDetails=Y'
            resp = requests.get(query_string)

#Get the latitude and longitude location of that address
            origin = []
            geo_location = json.loads(resp.content)
            if geo_location['found'] != 0:
                latitude = geo_location['results'][0]['LATITUDE']
                longitude = geo_location['results'][0]['LONGITUDE']
                origin.append((latitude, longitude))
                
# Append the Latitudes and Longitudes of the MRT Stations
            mrt_lat = mrt_location['latitude']
            mrt_long = mrt_location['longitude']
            list_of_mrt_coordinates = []
            for lat, long in zip(mrt_lat, mrt_long):
                list_of_mrt_coordinates.append((lat, long))

#Get distance to nearest MRT Stations (Mass Rapid Transit System)-----
            list_of_dist_mrt = []
            for destination in range(0, len(list_of_mrt_coordinates)):
                list_of_dist_mrt.append(geodesic(origin, list_of_mrt_coordinates[destination]).meters)
            shortest = (min(list_of_dist_mrt))
            min_dist_mrt = shortest
            list_of_dist_mrt.clear()
            st.write("Distance to Nearest MRT-Station",round(min_dist_mrt),"meters")
            
#Get distance from CDB (Central Business District)-----
            
            cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates
            st.write("Distance to Central Business District",round(cbd_dist),"meters")

#Input the user values to the model       
            new_sample = np.array(
                    [[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
            new_sample = scaler_loaded.transform(new_sample[:, :5])
            new_pred = loaded_model.predict(new_sample)[0]
            st.write('## :green[Predicted resale price:] ', round(np.exp(new_pred)))

    except Exception as e:
        st.write("Enter the above values to get the predicted resale price of the flat")  

#Menu-3-Insights
if selected=="Insights":
    st.header("Decision Tree")
    st.write("A decision tree is a non-parametric supervised learning algorithm for classification and regression tasks. It has a hierarchical tree structure consisting of a root node, branches, internal nodes, and leaf nodes. Decision trees are used for classification and regression tasks, providing easy-to-understand models")
    st.header("Insights")
    st.write("1.Resale Price depends on the Storey Range,Size of the Flat and Flat Type")
    st.write("2.Location near an MRT station and/or shopping mall and other amenities, it’s most likely to fetch a higher price")
    st.write("3.As the lease shortens, the resale value tends to decrease")
    st.write("4.The overall state of the property market can also affect HDB resale prices.")
    st.write("5.Additionally some factors such as Population,government policies,local market trends etc..,affect resale price")