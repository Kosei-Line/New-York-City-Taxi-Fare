import chainer, os
import pandas as pd
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import datasets, iterators, optimizers, serializers, training
from chainer.training import extensions
from sklearn.preprocessing import LabelEncoder

import Mod
args = Mod.args
Net = Mod.Net
def add_airport_dist(dataset):
    """
    Return minumum distance from pickup or dropoff coordinates to each airport.
    JFK: John F. Kennedy International Airport
    EWR: Newark Liberty International Airport
    LGA: LaGuardia Airport
    SOL: Statue of Liberty
    NYC: Newyork Central
    """
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892,-74.0445) # Statue of Liberty
    nyc_coord = (40.7141667,-74.0063889)


    pickup_lat = dataset['pickup_latitude']
    dropoff_lat = dataset['dropoff_latitude']
    pickup_lon = dataset['pickup_longitude']
    dropoff_lon = dataset['dropoff_longitude']

    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1])
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon)
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon)
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1])
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)
    pickup_sol = sphere_dist(pickup_lat, pickup_lon, sol_coord[0], sol_coord[1])
    dropoff_sol = sphere_dist(sol_coord[0], sol_coord[1], dropoff_lat, dropoff_lon)
    pickup_nyc = sphere_dist(pickup_lat, pickup_lon, nyc_coord[0], nyc_coord[1])
    dropoff_nyc = sphere_dist(nyc_coord[0], nyc_coord[1], dropoff_lat, dropoff_lon)



    dataset['jfk_dist'] = pickup_jfk + dropoff_jfk
    dataset['ewr_dist'] = pickup_ewr + dropoff_ewr
    dataset['lga_dist'] = pickup_lga + dropoff_lga
    dataset['sol_dist'] = pickup_sol + dropoff_sol
    dataset['nyc_dist'] = pickup_nyc + dropoff_nyc

    return dataset

def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon,
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))

def Load_Dataset():
    #csvからデータを読み取る
    data = pd.read_csv('test.csv')
    key = data[['key']]
    #必要なデータだけ取り出す
    #https://www.kaggle.com/code1110/houseprice-data-cleaning-visualization
    #必要なデータだけ取り出す
    data = data[['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude']]
    data = data.dropna(how='any', axis=0)
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    print('pickup_datetime')
    data['pickup_datetime_month'] = data['pickup_datetime'].dt.month
    data['pickup_datetime_year'] = data['pickup_datetime'].dt.year
    data['pickup_datetime_day_of_week'] = data['pickup_datetime'].dt.weekday
    data['pickup_datetime_day_of_hour'] = data['pickup_datetime'].dt.hour
    data = data.drop(['pickup_datetime'], axis = 1)
    print('in NY')
    data = data[(data['pickup_longitude'] > -75 )& (data['pickup_longitude'] < -72)]
    data = data[(data['dropoff_longitude'] > -75 )& (data['dropoff_longitude'] < -72)]
    data = data[(data['pickup_latitude'] > 39 )& (data['pickup_latitude'] < 43)]
    data = data[(data['dropoff_latitude'] > 39 )& (data['dropoff_latitude'] < 43)]
    print('distance')
    data['distance[km]'] = [sphere_dist(data.iloc[i,1], data.iloc[i,2], data.iloc[i,3],data.iloc[i,4]) for i in range(0, len(data['dropoff_latitude']))]
    add_airport_dist(data)
    data = data[(data['distance[km]'] > 0.01)]
    data['raised'] = 0
    data.loc[data['pickup_datetime_year'] > 2012, 'raised'] = 1
    data.loc[(data['pickup_datetime_month'] >8) & (data['pickup_datetime_year'] == 2012), 'raised'] = 1    #データを行列に変換
    test = data.as_matrix()
    key = key.as_matrix()
    #テストデータ
    test  = test[:,:].astype('float32')
    #テストデータを返す
    return test, key

def main():
        #学習ネットワークを持ってくる
    Rec = Net.Rec()
    #gpuを使う
    #CLS.to_gpu()
    #データセットの読み込み
    print('Loading dataset')
    test, key = Load_Dataset()
    print('Loaded dataset')
    a = []
    b = []
    serializers.load_npz('result/b{}/Rec_epoch_{}'.format(args.batch,
        args.epoch), Rec)
    for i in range(len(test)):
        with chainer.using_config('train', False):
            y = Rec(test[[i]]).data[0][0]
        b.append(y)
        a.append(key[i][0])
    df = pd.DataFrame({
        'Key' : a,
        'fare_amount' : b
    })
    df.to_csv("submit.csv",index=False)

if __name__ == '__main__':
    main()
