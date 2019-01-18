import os
import sys
import glob
import numpy as np
import pandas as pd
from IPython import embed

# Useage: python3 data/data_more_process.py ../data/modify_vldb_transport ../data

Freq = 1/30
miss_rate_thres = 0.15
WINDOW_SIZE = 130
SHIFT_SIZE = 128 # 128 128 64 64 256
Type2Shift_size_dict = {
    1: 128,
    2: 128,
    3: 64,
    4: 64,
    5: 256,
}

Type2Target_dict = {
    1: ['0', '7'],
    2: ['1'],
    3: ['2'],
    4: ['3'],
    5: ['4', '5', '6', '8', '9', '10', '12']
}

def Get_Array(a, SHIFT_SIZE):
    Dict, cou = {}, 0
    Total_count = int(a["Time"].max())
    l = a.values.tolist()
    # INT quantization
    for item in l:
        t = int(item[0])
        if t not in Dict.keys():
            Dict[t] = item[1:]
    # Fill the miss
    for i in range(Total_count+1):
        if i not in Dict.keys():
            cou+=1
            Dict[i] = Dict[i-1]
    # print("Miss rate: {}".format(cou/Total_count))

    # Generate time series data
    X, start = [], 0
    while start + WINDOW_SIZE - 1 <= Total_count:
        x = []
        for i in range(WINDOW_SIZE):
            x.append(Dict[i])
        X.append(x)
        start += SHIFT_SIZE
    return cou/Total_count, X

def process(filename, SHIFT_SIZE):
    # print((" "*100+filename)[-80:], end="\t")
    with open(filename) as f:
        a = f.read().strip("\n").split("\n") # Read
        a = np.array([line.split() for line in a]) # split line
        a = a[a[:,0].astype(int).argsort()] # Sort by time

        df = pd.DataFrame(a, columns=['Time', 'Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz'])
        df = df.replace("NAN", np.nan)
        df = df.astype(float)

        df["Time"] = df["Time"]/1e9 # change nanosecond to second
        lower_bound, upper_bound = df["Time"].min() + 15, df["Time"].max() - 15
        df = df.fillna(method='ffill') # fill na as previous data
        df = df.fillna(method='bfill') # fill na as next data
        df = df[df["Time"] >= lower_bound]
        df = df[df["Time"] <= upper_bound] # remove 15s error

        # print(filename)
        if df.isnull().values.any():
            return 2., None
            embed()
        if df.size != 0 :
            lower_bound, upper_bound = df["Time"].min(), df["Time"].max()
            # print("Minus: {}".format(upper_bound - lower_bound))
            # print("size: {}".format(df.size))

            df["Time"] = df["Time"] - lower_bound
            df["Time"] = round(df["Time"]/Freq)

            return Get_Array(df, SHIFT_SIZE)

        else:
            print(" ")
            return 1., None
        # else:      print("SIZE IS ZERO")
        # print("------------------------------------")


def GOGOGO(Type, Target, SHIFT_SIZE, origin_data_path, save_data_path):
    Dir1 = origin_data_path

    for f1_name in os.listdir(Dir1): # Train and Test
        writeNPY = "Type{}-GAM-{}".format(Type, f1_name.split("_")[2])
        print(writeNPY)
        DATA, NOISE = [], []
        # print(f1_name + "   --1")
        Dir2 = os.path.join(Dir1, f1_name)
        for f2_name in os.listdir(Dir2): # Every mode
            if f2_name.split("_")[0] in Target:
                # print(f2_name + "   --2")
                Dir3 = os.path.join(Dir2, f2_name)
                for f3_name in os.listdir(Dir3): # Every status
                    # print(f3_name + "   --3")
                    Dir4 = os.path.join(Dir3, f3_name)
                    for f4_name in os.listdir(Dir4): # Every log
                        # print(f4_name + "   --4")
                        miss_rate, out = process(os.path.join(Dir4, f4_name), SHIFT_SIZE)
                        if out != None and miss_rate < 0.5:
                            if miss_rate < miss_rate_thres:
                                if f1_name.split("_")[2] == "training":
                                    DATA.append(out)
                                else:
                                    DATA.append(out)
                            else:
                                    NOISE.append(out)
                        else:
                            print(miss_rate)
                        # else:
                        #     print(">>>>>>>>>>>>>{}__{}".format(os.path.join(Dir4, f4_name), miss_rate))

        DATA = [np.array(i) for i in DATA]
        NOISE = [np.array(i) for i in NOISE]

        DATA = np.array(DATA)
        NOISE = np.array(NOISE)

        print(writeNPY)
        np.save(os.path.join(save_data_path, writeNPY+".npy"), DATA)
        np.save(os.path.join(save_data_path, writeNPY+"_HMV.npy"), NOISE) # Higher Miss-rate Version
        
        assert np.array([ np.isnan(i).astype(int).sum()!=0 for i in DATA.tolist()]).astype(int).sum() == 0
        assert np.array([ np.isnan(i).astype(int).sum()!=0 for i in NOISE.tolist()]).astype(int).sum() == 0

def main():
    for Type in range(1, 6):
        Target = Type2Target_dict[Type]
        SHIFT_SIZE = Type2Shift_size_dict[Type]/4
        origin_data_path = sys.argv[1]
        save_data_path = sys.argv[2]
        GOGOGO(Type, Target, SHIFT_SIZE, origin_data_path, save_data_path)


if __name__ == '__main__':
    main()