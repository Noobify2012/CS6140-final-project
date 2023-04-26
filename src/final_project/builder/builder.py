"""builder.py
message pandas dataframes the way we like them to get messaged.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def cyclical_encode_dmy(df: pd.DataFrame) -> pd.DataFrame:
    cycleCats = ['DayofMonth', 'Month', 'Year']
    catsToCycleEncode = []
    for cat in cycleCats:
        pass
        if cat in df.columns:
            catsToCycleEncode.append(cat)
    if ('DayofMonth' in catsToCycleEncode):
        df['date_sin'] = np.sin(2 * np.pi * df['DayofMonth'] / 31)
        df['date_cos'] = np.cos(2 * np.pi * df['DayofMonth'] / 31)
    if ("Month" in catsToCycleEncode):
        df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    if ('Year' in catsToCycleEncode):
        df['year_sin'] = np.sin(2 * np.pi * df['Year'] / 100)
        df['year_cos'] = np.cos(2 * np.pi * df['Year'] / 100)
        print("length of vars to be dropped: " + str(len(catsToCycleEncode)))
    # return df
    if len(catsToCycleEncode) == 0:
        return df
    else:
        return df.drop(columns=catsToCycleEncode, axis=1)



def one_hot_encoding(
    df: pd.DataFrame, column_name: str, prefix_name: str
) -> pd.DataFrame:
    # enc = OneHotEncoder(handle_unknown='ignore').set_output(transform="pandas")
    # enc.fit(df)
    # print(enc.categories_)
    # encoded_frame = enc.transform(df)
    # print(encoded_frame)
    # encoded_frame = set_output
    return pd.get_dummies(df, columns=column_name, prefix=prefix_name)


def airlineAnalysis(frame: pd.DataFrame):
    frame["Operating_Airline"].value_counts().plot.barh(
        x="Number of Flights", y="Airline", rot=0
    )
    plt.show()


def originAnalysis(frame: pd.DataFrame, num: int):
    frame["Origin"].value_counts().iloc[:num].plot.barh(
        x="Number of Flights", y="Origin", rot=0
    )
    plt.show()


def destAnalysis(frame: pd.DataFrame, num: int):
    frame["Dest"].value_counts().iloc[:num].plot.barh(
        x="Number of Flights", y="Destination", rot=0
    )
    plt.show()


def delay15Analysis(frame: pd.DataFrame):
    frame["ArrDel15"].value_counts().plot.barh(
        x="Number of Flights", y="Delayed", rot=0
    )
    plt.show()


def delayAnalysis(frame: pd.DataFrame):
    #
    frame["ArrDel15"].value_counts().plot.barh(
        x="Number of Flights", y="Delayed", rot=0
    )
    plt.show()


def delayPlots(frame: pd.DataFrame):
    # filter out no delay data 0 min or less
    carrierFrame = frame.query("CarrierDelay > 0.0")
    weatherFrame = frame.query("WeatherDelay > 0.0")
    nasFrame = frame.query("NASDelay > 0.0")
    securityFrame = frame.query("SecurityDelay > 0.0")
    laFrame = frame.query("LateAircraftDelay > 0.0")
    # frame = frame.query('CarrierDelay > 0.0 | WeatherDelay > 0.0 | NASDelay > 0.0 | SecurityDelay > 0.0 | LateAircraftDelay > 0.0')
    carrierFrame["CarrierDelay"].value_counts().iloc[:20].plot.barh(
        x="Count", y="Delay Time"
    )
    weatherFrame["WeatherDelay"].value_counts().iloc[:20].plot.barh(
        x="Count", y="Delay Time"
    )
    nasFrame["NASDelay"].value_counts().iloc[:20].plot.barh(
        x="Count", y="Delay Time"
    )
    securityFrame["SecurityDelay"].value_counts().iloc[:20].plot.barh(
        x="Count", y="Delay Time"
    )
    laFrame["LateAircraftDelay"].value_counts().iloc[:20].plot.barh(
        x="Count", y="Delay Time"
    )
    # frame['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'].value_counts().plot.barh()
    plt.show()


def boxplotData(frame: pd.DataFrame):
    # df = pd.melt(frame['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'])
    boxplot = frame.boxplot(
        column=[
            "CarrierDelay",
            "WeatherDelay",
            "NASDelay",
            "SecurityDelay",
            "LateAircraftDelay",
        ],
        rot=45,
    )
    plt.show()


def delayCounts(frame: pd.DataFrame):
    mean_carrier_delay = frame["CarrierDelay"].mean()
    mean_weather_delay = frame["WeatherDelay"].mean()
    mean_nas_delay = frame["NASDelay"].mean()
    mean_security_delay = frame["SecurityDelay"].mean()
    mean_la_delay = frame["LateAircraftDelay"].mean()
    print(
        "Mean Carrier Delay: "
        + str(mean_carrier_delay)
        + "\nMean Weather Delay: "
        + str(mean_weather_delay)
        + "\nMean National Airspace Delay: "
        + str(mean_nas_delay)
        + "\nMean Security Delay: "
        + str(mean_security_delay)
        + "\nMean Late Aircraft Delay: "
        + str(mean_la_delay)
    )
    median_carrier_delay = frame["CarrierDelay"].median()
    median_weather_delay = frame["WeatherDelay"].median()
    median_nas_delay = frame["NASDelay"].median()
    median_security_delay = frame["SecurityDelay"].median()
    median_la_delay = frame["LateAircraftDelay"].median()
    print(
        "Median Carrier Delay: "
        + str(median_carrier_delay)
        + "\nMedian Weather Delay: "
        + str(median_weather_delay)
        + "\nMedian National Airspace Delay: "
        + str(median_nas_delay)
        + "\nMedian Security Delay: "
        + str(median_security_delay)
        + "\nMedian Late Aircraft Delay: "
        + str(median_la_delay)
    )


def DistanceAnalysis(frame: pd.DataFrame, num: int):
    frame["DistanceGroup"].value_counts().iloc[:num].plot.barh(
        x="Number of Flights", y="Origin", rot=0
    )
    plt.show()


def delayPercentage(frame: pd.DataFrame):
    delayed = frame["ArrDel15"].value_counts()[1.0]
    notDel = frame["ArrDel15"].value_counts()[0.0]
    total = delayed + notDel
    percentage = (delayed / notDel) * 100
    # print('value of delayed: ' + str(delayed))
    # print('value of total: ' + str(total))
    print("Percentage of flights delayed: " + str(percentage))


def runEDA(df: pd.DataFrame) -> pd.DataFrame:
    # Show the number of flights by airline
    airlineAnalysis(df)
    # Show the number of flights by origin
    originAnalysis(df, 20)
    # show the number of flights by destination
    destAnalysis(df, 20)
    # duplicated method for number of flights delayed
    # delay15Analysis(df)
    # show the plots of all the differnt delay types
    delayPlots(df)
    # show all of the delay types on a single boxplot
    boxplotData(df)
    # get the mean and median of the differnt types of delay
    delayCounts(df)
    # show the number of flights by distance group
    DistanceAnalysis(df, 20)
    # Calculate percentage of flights that are delayed
    delayPercentage(df)


# one hot encode origin and destination
def oneHotEncoding(df:pd.DataFrame, columnName:list, prefixName:str):
    enc = pd.get_dummies(df, columns=columnName, prefix=prefixName)
    return enc


def encodeFrame(frame: pd.DataFrame):
    # Cyclically encode the day, month, and year
    frame = cyclical_encode_dmy(frame)
    # test_frame_encoded.head(5)
    #encode the operating airline, origin, and destination codes
    hotencodedCats = ['Operating_Airline', 'Origin', 'Dest']
    hotcatsToEncode = []
    
    #see what frames to encode
    for cats in hotencodedCats:
        if cats in frame.columns:
            hotcatsToEncode.append(cats)
    
    for cat in hotcatsToEncode:
        frame = oneHotEncoding(frame, [cat], cat)
    

    #Drop the duplicate category
    print(frame.columns)
    frame.drop(['Duplicate'], axis=1, inplace=True)
    #fill the delay type NANs
    frame = frame.fillna(0)
    # print(frame.head(5))
    return frame
    # test_frame_encoded.head(5)

def columnManager(frame):
    # frame2 = frame[['DayofMonth',  'Origin', 'Operating_Airline','ArrDel15', 'DistanceGroup', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Duplicate']].copy()
    frame = frame[['Year', 'Month', 'DayofMonth', 'Operating_Airline', 'Origin', 'Dest', 'ArrDel15', 'DistanceGroup', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Duplicate']].copy()

    return frame



