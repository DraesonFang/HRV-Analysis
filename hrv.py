import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astroid import NotFoundError
from dask.multiprocessing import exceptions
from sqlalchemy.dialects.mssql.information_schema import columns


def calculate_HRV_metrics(file_in):
    # your code here
    results = {"filename": file_in,
               "n": None,
               "mean_nn": None,
               "mean_bpm": None,
               "sdnn": None,
               "rmssd": None,
               "pnn20": None,
               "pnn50": None}
    try:
        data = pd.read_csv(file_in)
        df = pd.DataFrame(data)
        df['time_next'] = df['time'].shift(-1)
        df['type_next'] = df['type'].shift(-1)
        df['rr'] = df['time_next'] - df['time']
        df['rr_type'] = df['type'] + df['type_next']
        df['rr_next'] = df['rr'].shift(-1)
        df['diff'] = df['rr_next'] - df['rr']
        df['diff_squared'] = df['diff'] ** 2
        df['rr_next_type'] = df['rr_type'].shift(-1)

        results['filename'] = file_in.split('/')[1]
        results['n'] = df[df['rr_type']=='NN'].shape[0] if df[df['type']=='N'].shape[0] >= 500 else None
        results['mean_nn'] = round(df['rr'].mean()) if df['rr'].shape[0] >= 500 else None
        results['mean_bpm'] = round(60000 / results['mean_nn'], 1) if df['rr'].shape[0] >= 500 else None
        results['sdnn'] = round(df[df['rr_type'] == 'NN']['rr'].std(),1) if df['rr'].shape[0] >= 500 else None
        results['rmssd'] = round(np.sqrt(df.loc[(df['rr_type'] =='NN') & (df['rr_type'].shift(-1) == 'NN'),'diff_squared'].mean()),1) if df.loc[(df['rr_type'] =='NN') & (df['rr_type'].shift(-1) == 'NN'),'diff_squared'].shape[0] >= 500 else None
        results['pnn20'] = round((np.abs(df.loc[(df['rr_type'] =='NN') & (df['rr_type'].shift(-1) == 'NN'),'diff'])>20).sum()/(results['n']-1)*100,1) if df.loc[(df['rr_type'] =='NN') & (df['rr_type'].shift(-1) == 'NN'),'diff'].shape[0] >= 500 and df.shape[0] >= 500 else None
        results['pnn50'] = round((np.abs(df.loc[(df['rr_type'] =='NN') & (df['rr_type'].shift(-1) == 'NN'),'diff'])>50).sum()/(results['n']-1)*100,1) if df.loc[(df['rr_type'] =='NN') & (df['rr_type'].shift(-1) == 'NN'),'diff'].shape[0] >= 500 and df.shape[0] >= 500 else None

        return results
    except FileNotFoundError:
        print("file not found")



def process_HRV_files(file_list_in, file_out):
    # your code here
    columns_name = ['filename', "mean_nn", 'mean_bpm', 'sdnn', 'rmssd', 'pnn20', 'pnn50']
    df = pd.DataFrame(columns=columns_name)

    try:
        for file in file_list_in:
            try:
                if os.path.exists(file):
                    temp_df = pd.DataFrame([calculate_HRV_metrics(file)])
                    temp_df = temp_df[df.columns]
                    df = pd.concat([df,temp_df], ignore_index=True)
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                print(f"File '{file}' does not exist")
        df.to_csv(file_out, index=False)
    except Exception as e:
            print(e)
    return file_out

print(calculate_HRV_metrics('investigating_hrv_dataset/y01.csv'))
