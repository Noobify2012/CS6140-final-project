#!/usr/bin/env python

#Authors: Dan Blum, Matt Greene, Bram Zilzter
#Dataset: https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022?select=readme.md

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# add dataframe with Distance included


def readfile(path):
    #read csv into data frame
    init_frame = pd.read_csv(path, engine='pyarrow')
    #grab columns we need
    #Year, Month, DayofMonth, Operating_Airline, Origin, Dest, ArrDel15, DistanceGroup, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay, Duplicate
    frame = init_frame[['Year', 'Month', 'DayofMonth', 'Operating_Airline ', 'Origin', 'Dest', 'ArrDel15', 'DistanceGroup', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Duplicate']].copy()
    #get rid of space in operating airline
    frame.rename(columns={"Operating_Airline ": "Operating_Airline"})
    return frame


def airlineAnalysis(frame):
    frame['Operating_Airline '].value_counts().plot.barh(x='Number of Flights', y='Airline', rot=0)
    plt.show();


def originAnalysis(frame, num):
    frame['Origin'].value_counts().iloc[:num].plot.barh(x='Number of Flights', y='Origin', rot=0)
    plt.show();



def destAnalysis(frame, num):
    frame['Dest'].value_counts().iloc[:num].plot.barh(x='Number of Flights', y='Destination', rot=0)
    plt.show();


def delay15Analysis(frame):
    frame['ArrDel15'].value_counts().plot.barh(x='Number of Flights', y='Delayed', rot=0)
    plt.show();


def delayAnalysis(frame):
    #
    frame['ArrDel15'].value_counts().plot.barh(x='Number of Flights', y='Delayed', rot=0)
    plt.show();


def delayPlots(frame):
    # filter out no delay data 0 min or less
    carrierFrame = frame.query('CarrierDelay > 0.0')
    weatherFrame = frame.query('WeatherDelay > 0.0')
    nasFrame = frame.query('NASDelay > 0.0')
    securityFrame = frame.query('SecurityDelay > 0.0')
    laFrame = frame.query('LateAircraftDelay > 0.0')
    # frame = frame.query('CarrierDelay > 0.0 | WeatherDelay > 0.0 | NASDelay > 0.0 | SecurityDelay > 0.0 | LateAircraftDelay > 0.0')
    carrierFrame['CarrierDelay'].value_counts().iloc[:20].plot.barh(x="Count", y="Delay Time")
    weatherFrame['WeatherDelay'].value_counts().iloc[:20].plot.barh(x="Count", y="Delay Time")
    nasFrame['NASDelay'].value_counts().iloc[:20].plot.barh(x="Count", y="Delay Time")
    securityFrame['SecurityDelay'].value_counts().iloc[:20].plot.barh(x="Count", y="Delay Time")
    laFrame['LateAircraftDelay'].value_counts().iloc[:20].plot.barh(x="Count", y="Delay Time")
    # frame['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'].value_counts().plot.barh()
    plt.show();


def boxplotData(frame):
    # df = pd.melt(frame['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'])
    boxplot = frame.boxplot(column=['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'], rot=45)
    plt.show()


def delayCounts(frame):
    mean_carrier_delay = frame['CarrierDelay'].mean()
    mean_weather_delay = frame['WeatherDelay'].mean()
    mean_nas_delay = frame['NASDelay'].mean()
    mean_security_delay = frame['SecurityDelay'].mean()
    mean_la_delay = frame['LateAircraftDelay'].mean()
    print("Mean Carrier Delay: " + str(mean_carrier_delay) + "\nMean Weather Delay: " + str(mean_weather_delay) + "\nMean National Airspace Delay: " + str(mean_nas_delay) + "\nMean Security Delay: " + str(mean_security_delay) + "\nMean Late Aircraft Delay: " + str(mean_la_delay))
    median_carrier_delay = frame['CarrierDelay'].median()
    median_weather_delay = frame['WeatherDelay'].median()
    median_nas_delay = frame['NASDelay'].median()
    median_security_delay = frame['SecurityDelay'].median()
    median_la_delay = frame['LateAircraftDelay'].median()
    print("Median Carrier Delay: " + str(median_carrier_delay) + "\nMedian Weather Delay: " + str(median_weather_delay) + "\nMedian National Airspace Delay: " + str(median_nas_delay) + "\nMedian Security Delay: " + str(median_security_delay) + "\nMedian Late Aircraft Delay: " + str(median_la_delay))



def DistanceAnalysis(frame, num):
    frame['DistanceGroup'].value_counts().iloc[:num].plot.barh(x='Number of Flights', y='Origin', rot=0)
    plt.show();



def delayPercentage(frame):
    delayed = frame['ArrDel15'].value_counts()[1.0]
    notDel = frame['ArrDel15'].value_counts()[0.0]
    total = delayed + notDel
    percentage = (delayed/notDel) * 100
    # print('value of delayed: ' + str(delayed))
    # print('value of total: ' + str(total))
    print('Percentage of flights delayed: ' + str(percentage))



def cyclicalEncodeDMY(df):
    df['date_sin'] = np.sin(2*np.pi*df['DayofMonth']/31)
    df['date_cos'] = np.cos(2*np.pi*df['DayofMonth']/31)
    df['month_sin'] = np.sin(2*np.pi*df['Month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['Month']/12)
    df['year_sin'] = np.sin(2*np.pi*df['Year']/100)
    df['year_cos'] = np.cos(2*np.pi*df['Year']/100)
    df.drop(['DayofMonth', 'Month', 'Year'], axis=1, inplace=True)
    return df


# one hot encode origin and destination
def oneHotEncoding(df, columnName, prefixName):
    enc = pd.get_dummies(df, columns=columnName, prefix=prefixName)
    return enc

#Perform EDA
# get percentage of delays 
# get number of delays when not 0


def buildDataFrame(path):
    #read the files
    df = pd.DataFrame()
    test = 0;
    print("Size of df at init:  " + str(df.size))
    for filename in os.listdir(path):
        file = os.path.join(path,filename)
        if os.path.isfile(file):
            print(filename)
            frame = readfile(file)
            # pd.concat([df, frame])
            print("the frame i got back: ")
            print(frame.head(5))
            if (df.size == 0):
                df = frame
            else: 
                pd.concat([df, frame], columns=['Year', 'Month', 'DayofMonth', 'Operating_Airline ', 'Origin', 'Dest', 'ArrDel15', 'DistanceGroup', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Duplicate'])
                # df.append(frame)
            print("Size of df: " + str(df.size))
            if test > 2:
                break
            test += 1
    return df

def runEDA(frame):
    # print(test_frame.iloc[0])
    # Whats the percentage of flights that are delayed based on ArrDel15
    delayPercentage(frame)
    # Show the number of flights by airline
    airlineAnalysis(frame)
    # Show the number of flights by origin
    originAnalysis(frame, 20)
    #show the number of flights by destination
    destAnalysis(frame, 20)
    # duplicated method for number of flights delayed
    delay15Analysis(frame)
    # show the plots of all the differnt delay types
    delayPlots(frame)
    # show all of the delay types on a single boxplot
    boxplotData(frame)
    # get the mean and median of the differnt types of delay
    delayCounts(frame)
    # show the number of flights by distance group
    DistanceAnalysis(frame, 20)
    # Calculate percentage of flights that are delayed
    delayPercentage(frame)

def encodeFrame(frame):
    #Cyclically encode the day, month, and year
    frame_encoded = cyclicalEncodeDMY(frame)
    # test_frame_encoded.head(5)
    #encode the operating airline, origin, and destination codes
    catsToEncode = ['Operating_Airline ', 'Origin', 'Dest']
    for cat in catsToEncode:
        frame_encoded = oneHotEncoding(frame_encoded, [cat], cat)
    print(frame_encoded.head(5))

    #Drop the duplicate category
    frame_encoded.drop(['Duplicate'], axis=1, inplace=True)
    #fill the delay type NANs
    frame_encoded = frame_encoded.fillna(0)
    # test_frame_encoded.head(5)

#set the file path
path = '../../raw/'
# pwd = "pwd"
# os.system(pwd)
# cd = "cd ../../"
# ls = "ls"
# os.system(cd)
# os.system(ls)
# os.system(pwd)
test_frame = buildDataFrame(path)
print(test_frame.head(10))
print(test_frame.size)