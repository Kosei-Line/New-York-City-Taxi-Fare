#https://www.kaggle.com/namakaho/nyctaxi
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
Evaluator = Mod.Evaluator
Updater = Mod.Updater
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
    data = pd.read_csv('train.csv', nrows=1000)
    #必要なデータだけ取り出す
    data = data[['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude', 'fare_amount']]
    #欠損データにさようなら
    data = data.dropna(how='any', axis=0)
    #日時を取得
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    print('pickup_datetime')
    data['pickup_datetime_month'] = data['pickup_datetime'].dt.month
    data['pickup_datetime_year'] = data['pickup_datetime'].dt.year
    data['pickup_datetime_day_of_week'] = data['pickup_datetime'].dt.weekday
    data['pickup_datetime_day_of_hour'] = data['pickup_datetime'].dt.hour
    data = data.drop(['pickup_datetime'], axis = 1)\
    #NY近郊か
    print('in NY')
    data = data[(data['pickup_longitude'] > -75 )& (data['pickup_longitude'] < -72)]
    data = data[(data['dropoff_longitude'] > -75 )& (data['dropoff_longitude'] < -72)]
    data = data[(data['pickup_latitude'] > 39 )& (data['pickup_latitude'] < 43)]
    data = data[(data['dropoff_latitude'] > 39 )& (data['dropoff_latitude'] < 43)]
    #移動距離
    print('distance')
    data['distance[km]'] = [sphere_dist(data.iloc[i,1], data.iloc[i,2], data.iloc[i,3],data.iloc[i,4]) for i in range(0, len(data['dropoff_latitude']))]
    #空港
    add_airport_dist(data)
    #常識的な料金と距離
    data = data[(data['fare_amount'] > 0.01)]
    data = data[(data['distance[km]'] > 0.01)]
    #2012年の値上げ
    data['raised'] = 0
    data.loc[data['pickup_datetime_year'] > 2012, 'raised'] = 1
    data.loc[(data['pickup_datetime_month'] >8) & (data['pickup_datetime_year'] == 2012), 'raised'] = 1    #データを行列に変換
    T = data['fare_amount']
    X = data.drop(['fare_amount'], axis=1)
    print(X.columns)
    pria
    X = X.as_matrix().astype(np.float32)
    T = T.as_matrix().astype(np.float32)
    T = np.reshape(T,[-1, 1])
    #訓練データとテストデータに分ける
    thresh_hold = int(X.shape[0]*0.8)
    train = datasets.TupleDataset(X[:thresh_hold], T[:thresh_hold])
    test  = datasets.TupleDataset(X[thresh_hold:], T[thresh_hold:])
    #訓練データとテストデータを返す
    return train, test

def main():
    #学習ネットワークを持ってくる
    Rec = Net.Rec()
    #gpuを使う
    Rec.to_gpu()
    #データセットの読み込み
    print('Loading dataset')
    train, test = Load_Dataset()
    print('Loaded dataset')

    #make_optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.9, beta2=0.999):
        optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
        return optimizer
    opt = make_optimizer(Rec)
    #set iterator
    train_iter = iterators.SerialIterator(train, args.batch)
    test_iter  = iterators.SerialIterator(test, args.batch,
        repeat=False, shuffle=False)
    #define updater
    updater = Updater.MyUpdater(train_iter, Rec, opt, device=args.gpu)
    #define trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'),
        out="{}/b{}".format(args.out, args.batch))
    #define evaluator
    trainer.extend(Evaluator.MyEvaluator(test_iter, Rec, device=args.gpu))
    #save model
    trainer.extend(extensions.snapshot_object(Rec,
        filename='Rec_epoch_{.updater.epoch}'),
        trigger=(args.snapshot, 'epoch'))
    #out Log
    trainer.extend(extensions.LogReport())
    #print Report
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'val/loss', 'elapsed_time']))
    #display Progress bar
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    del trainer


if __name__ == '__main__':
    main()
    #train, test = Load_Dataset()
