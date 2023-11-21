#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from word2number import w2n
import matplotlib
import warnings
warnings.filterwarnings("ignore")

# convert column's dtype from an object to datetime
def datCnv(src):
    return pd.to_datetime(src)

# checks if column contains number in string  
def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

# convert numbers written in alphabetical characeters to a numberic value 
def convert_word_to_number(word):
    word = word.lower()
    number = w2n.word_to_num(word)
    return number 

#CALCULATING PROFIT,COSTS,REVENUE
def calculate_profit(agg):
    #arrival and departure delay costs on the first half of the round trip flight
    agg['DEP_DELAY_mean_cost'] = (agg['DEP_DELAY_mean'] - 15)*75
    agg['DEP_DELAY_mean_cost'] = np.where(agg.DEP_DELAY_mean_cost < 0, 0, agg.DEP_DELAY_mean_cost)
    agg['ARR_DELAY_mean_cost'] = (agg['ARR_DELAY_mean'] - 15)*75
    agg['ARR_DELAY_mean_cost'] = np.where(agg.ARR_DELAY_mean_cost < 0, 0, agg.ARR_DELAY_mean_cost)
    
    #arrival and departure delay costs on second half of the round trip flight
    agg['NEXT_DEP_DELAY_mean_cost'] = (agg['NEXT_DEP_DELAY_mean'] - 15)*75
    agg['NEXT_DEP_DELAY_mean_cost'] = np.where(agg.NEXT_DEP_DELAY_mean_cost < 0, 0, agg.NEXT_DEP_DELAY_mean_cost)
    agg['NEXT_ARR_DELAY_mean_cost'] = (agg['NEXT_ARR_DELAY_mean'] - 15)*75
    agg['NEXT_ARR_DELAY_mean_cost'] = np.where(agg.NEXT_ARR_DELAY_mean_cost < 0, 0, agg.NEXT_ARR_DELAY_mean_cost)
    
    #other sources of costs and revenue based on the assumptions given
    agg['FOMC_cost'] = agg['DISTANCE_mean']*8
    agg['DIO_cost'] = agg['DISTANCE_mean']*1.18
    agg['NO_PASSENGERS_mean'] = (agg['OCCUPANCY_RATE_mean']*200).astype(int)
    agg['BAGGAGE_mean_revenue'] = agg['NO_PASSENGERS_mean']*0.5*70
    agg['ITIN_mean_revenue'] = agg['NO_PASSENGERS_mean']* agg['ITIN_FARE_median']
    agg['TOTAL_REVENUE_mean'] = agg['BAGGAGE_mean_revenue'] + agg['ITIN_mean_revenue']
    agg['TOTAL_operational_cost'] = agg['ORIGIN_op_cost'] + agg['DESTINATION_op_cost']
    agg['TOTAL_COST_mean'] = agg['DEP_DELAY_mean_cost'] + agg['ARR_DELAY_mean_cost'] + agg['TOTAL_operational_cost'] + agg['FOMC_cost'] + agg['NEXT_DEP_DELAY_mean_cost'] +agg['NEXT_ARR_DELAY_mean_cost']
    agg['TOTAL_PROFIT_mean'] = agg['TOTAL_REVENUE_mean'] - agg['TOTAL_COST_mean']
    agg = agg.sort_values(by='TOTAL_PROFIT_mean', ascending=False)
    return agg


#include only round trip flights using the tickets dataset
def process_tickets_dataset(dataset):
    dataset = dataset[dataset.ROUNDTRIP == 1.0]
    dataset['ITIN_FARE'] = dataset['ITIN_FARE'].str.replace("$","").astype(float)
    dataset = dataset[['ORIGIN','DESTINATION','ITIN_FARE']]
    return dataset

#process flights dataset
def process_flights(flights,airport_codes):
    #exclude canceled flights and drop the cancelled column
    flights = flights.drop(columns=['OP_CARRIER','ORIGIN_AIRPORT_ID','OP_CARRIER_FL_NUM','ORIGIN_CITY_NAME', 'DEST_AIRPORT_ID','DEST_CITY_NAME','AIR_TIME'],axis=1)
    flights = flights[flights['CANCELLED'] == 0.0].drop(columns=['CANCELLED'],axis=1)

    #classify airport size of the airport of the columns ORIGIN and Destination of the flights dataset
    airport_codes = airport_codes.rename(columns={"IATA_CODE": "ORIGIN"})
    flights =  pd.merge(flights,airport_codes,how='inner',on='ORIGIN')
    airport_codes = airport_codes.rename(columns={"ORIGIN": "DESTINATION"})
    flights =  pd.merge(flights,airport_codes,how='inner',on='DESTINATION')
    flights = flights.rename(columns={"TYPE_x": "ORIGIN_TYPE",'TYPE_y':"DESTINATION_TYPE"})

    #convert FL_DATE column into datetime64[ns] Dtype
    flights['FL_DATE'] = flights.FL_DATE.apply(datCnv)

    #sort flights dataset by ascending order based on the date(FL_DATE)
    flights = flights.sort_values(by='FL_DATE',ascending=True)
    flights['DISTANCE']  = flights['DISTANCE'].replace(['****','NAN'], np.nan)
    flights['has_number'] = flights.apply(lambda x: has_numbers(str(x.DISTANCE)), axis=1)
    distance_words = list(np.unique(list(flights[~flights['has_number']]['DISTANCE'].dropna())))
    for word in distance_words:
        flights['DISTANCE']  = flights['DISTANCE'].replace(word, convert_word_to_number(word)) 
    flights['DISTANCE'] = flights['DISTANCE'].astype(float)
    flights['DISTANCE'] = round(2*flights['DISTANCE'])
    flights['starts_with_N'] = list(map(lambda x: x.startswith('N'), flights['TAIL_NUM']))     
    flights = flights[flights['starts_with_N']].drop(columns=['has_number','starts_with_N'])
    flights['ORIGIN_op_cost'] = np.where(flights['ORIGIN_TYPE'] == 'medium_airport',5000,10000)
    flights['DESTINATION_op_cost'] = np.where(flights['DESTINATION_TYPE'] == 'medium_airport',5000,10000)
    flights = flights.drop(columns=['ORIGIN_TYPE','DESTINATION_TYPE'])
    return flights

def filter_flights(flights):
    table = pd.DataFrame(columns = flights.columns)
    #list of the Tail Number of each unique plane 
    unique_plane_codes = list(np.unique(list(flights['TAIL_NUM'])))
    for plane_code in unique_plane_codes:
        #include round trip flights per unique plane using the TAIL
        flights_plane = flights[flights['TAIL_NUM']==plane_code]
        flights_shifted = flights_plane.shift(-1)

        #calculate delay time on the way back of the round trip
        flights_plane['NEXT_ARR_DELAY'] = flights_shifted['ARR_DELAY']
        flights_plane['NEXT_DEP_DELAY'] = flights_shifted['DEP_DELAY']

        #include only round trips
        flights_plane['NEXT_ORIGIN'] = flights_shifted['ORIGIN']
        flights_plane['NEXT_DESTINATION'] = flights_shifted['DESTINATION']
        #conditions_1 and condition_2 are based on the assumption stated above
        condition_1 = (flights_plane['NEXT_DESTINATION'] == flights_plane['ORIGIN'] ).astype(int)
        condition_2 = (flights_plane['NEXT_ORIGIN'] == flights_plane['DESTINATION']).astype(int)
        flights_plane['CON_1'] = condition_1
        flights_plane['CON_2'] = condition_2
        flights_plane['ROUND_TRIP'] = (flights_plane['CON_1'].astype(int) + flights_plane['CON_2'].astype(int))
        #if ROUND_TRIP equals to 2, the flight is a round trip
        flights_plane = flights_plane[flights_plane['ROUND_TRIP']==2]

        #drop irrelevant columns
        flights_plane = flights_plane.drop(columns=['NEXT_ORIGIN','NEXT_DESTINATION','CON_1', 'CON_2','ROUND_TRIP'],axis=1) 
        table = pd.concat([table,flights_plane])
    return table


#the 3 datasets (airports,flights,tickets) are stored in a folder named data in the previous directory
destination_folder = '../data/'

#reading airports dataset
airport_codes = pd.read_csv( destination_folder + 'airport_codes.csv')[['TYPE','IATA_CODE']]

#drop rows where IATA_CODE is a NaN value
airport_codes = airport_codes[airport_codes['IATA_CODE'].notnull()]

#filtering for only medium and large size airports
airport_codes = airport_codes[(airport_codes['TYPE'] == 'medium_airport') | (airport_codes['TYPE'] == 'large_airport')]

#read flights dataset, and drop irrelevant columns
flights = pd.read_csv(destination_folder + 'Flights.csv')
flights = process_flights(flights,airport_codes)
df = filter_flights(flights)
df.to_csv('flights_filtered.csv')

#read filtered flights dataset
df = pd.read_csv('flights_filtered.csv',index_col=[0])

#count number of flights per round trip combination (
agg1 = df.groupby(['ORIGIN', 'DESTINATION']).size().reset_index().rename(columns={0:'NO_FLIGHTS'})

#calculating the mean of variables that contribute to the calculation of the total profit
means1 = df.groupby(['ORIGIN', 'DESTINATION']).agg({'DISTANCE': 'mean','NEXT_ARR_DELAY':'mean','NEXT_DEP_DELAY':'mean','DEP_DELAY': 'mean','ARR_DELAY': 'mean','OCCUPANCY_RATE': 'mean', 'ORIGIN_op_cost':'mean','DESTINATION_op_cost':'mean'}).reset_index()

#read and process tickets dataset
tickets = pd.read_csv( destination_folder + 'tickets.csv')
tickets = process_tickets_dataset(tickets)

#aggregate median value for itinerary fare for each round trip route combintion
itin_fare_median1 = tickets.groupby(['ORIGIN','DESTINATION'])['ITIN_FARE'].apply('median').reset_index().dropna()
itin_fare_median1['ORIGIN_DESTINATION'] = itin_fare_median1['ORIGIN'] + '_' + itin_fare_median1['DESTINATION']


#aggregate means and ticket price for each round trip route combination
agg1 = pd.merge(agg1,means1,on=['ORIGIN','DESTINATION'], how='left')
agg1 = pd.merge(agg1,itin_fare_median1,on=['ORIGIN','DESTINATION'], how='left').rename(columns={'DISTANCE':'DISTANCE_mean','DEP_DELAY':'DEP_DELAY_mean','ARR_DELAY':'ARR_DELAY_mean','OCCUPANCY_RATE':'OCCUPANCY_RATE_mean','ITIN_FARE':'ITIN_FARE_median','NEXT_ARR_DELAY':'NEXT_ARR_DELAY_mean','NEXT_DEP_DELAY':'NEXT_DEP_DELAY_mean' }).dropna()
agg1 = calculate_profit(agg1)

agg1['FACTOR'] = agg1['TOTAL_PROFIT_mean'] * agg1['NO_FLIGHTS']
agg1 = agg1.sort_values(by='FACTOR', ascending=False)
agg1_recc = agg1[:10]

upfront_airplane_cost = 90*(10**6)
agg1_recc['NO_FLIGHTS_BREAKEVEN'] = (upfront_airplane_cost/agg1_recc['TOTAL_PROFIT_mean']).astype(int)
agg1_recc['KPI'] = np.log(1+agg1_recc.FACTOR)
df_summary = agg1_recc[['NO_FLIGHTS','ORIGIN_DESTINATION','TOTAL_REVENUE_mean','TOTAL_COST_mean','TOTAL_PROFIT_mean','NO_FLIGHTS_BREAKEVEN','KPI']]
df_summary.to_csv('df_summary.csv')

