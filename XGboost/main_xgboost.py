import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import time
import os
import argparse

from utils import models2 as models
from sys import stdout
from IPython import embed
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
import pickle
import sys

device = "cuda" if torch.cuda.is_available else "cpu"

class main():
    def __init__(self, args):
        # args
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr_rate
        self.path = args.data_path
        self.print_iter = args.print_iter
        self.save_iter = args.save_iter
        self.model_dump = args.model_dump

        # init
        self.loss_epoch_tr = []
        self.acc_epoch_tr = []
        self.loss_epoch_va = []
        self.acc_epoch_va = []


    def load_data(self):
        print("Load prepro-data...")
        Trans = []

        train_data_list = [
                           "Type1-GAM-training.npy",
                           "Type2-GAM-training.npy",
                           "Type3-GAM-training.npy",
                           "Type4-GAM-training.npy",
                           "Type5-GAM-training.npy"]

        train_x, valid_x = [], []
        train_y, valid_y = [], []
        a = {0:[],1:[],2:[],3:[],4:[]}
        for idx, file in enumerate(train_data_list):
            data = np.load(os.path.join(self.path, file))
            # assert np.array([ np.isnan(i).astype(int).sum()!=0 for i in data.tolist()]).astype(int).sum() == 0
            data_tr, data_va = train_test_split(data.tolist(), test_size=0.2, random_state=1126)

            print("\tThere are {} ({}+{}) log files in {} data with label {}.".format(len(data_tr)+len(data_va), len(data_tr), len(data_va), file, idx))
            tmp_x = []

            for file in data_tr:
                for row in file:
                    flat_row = row.reshape(-1)
                    tmp_x.append(flat_row)

            if idx == 4:
                tmp_x = random.sample(tmp_x, 50000)
            tmp_y = [idx]*len(tmp_x)

            train_x.extend(tmp_x)
            train_y.extend(tmp_y)
            print("\t\tThere are {} train data.".format(len(tmp_x)))

            tmp_x = []
            for file in data_va:
                for row in file:
                    flat_row = row.reshape(-1)
                    tmp_x.append(flat_row)

            if idx == 4:
                tmp_x = random.sample(tmp_x, 18000)
            tmp_y = [idx]*len(tmp_x)
            valid_x.extend(tmp_x)
            valid_y.extend(tmp_y)
            print("\t\tThere are {} valid data.".format(len(tmp_x)))

        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.valid_x = np.array(valid_x)
        self.valid_y = np.array(valid_y)
        self.train_num_x = len(self.train_x)
        self.valid_num_x = len(self.valid_x)
        print("There are {} training data.".format(self.train_num_x))
        print("There are {} validation data.".format(self.valid_num_x))


    def load_data_aug(self):
        print("Load prepro-data...")
        Trans = []
        while len(Trans) < 64:
            a, b = np.random.uniform(low=-100, high=100, size=(3,)), np.random.uniform(low=-100, high=100, size=(3,))
            while np.linalg.norm(a) < 0.1  :
                a = np.random.uniform(low=-100, high=100, size=(3,))
            while np.linalg.norm(b) < 0.1  :
                b = np.random.uniform(low=-100, high=100, size=(3,))
            c = np.cross(a, b)
            b = np.cross(a, c)

            a = (a/np.linalg.norm(a)).reshape(-1, 1)
            b = (b/np.linalg.norm(b)).reshape(-1, 1)
            c = (c/np.linalg.norm(c)).reshape(-1, 1)
            trans = np.concatenate((a, b, c), axis=1)
            Trans.append(trans)
        self.Trans = Trans

        train_data_list = [
                           "Type1-GAM-training.npy",
                           "Type2-GAM-training.npy",
                           "Type3-GAM-training.npy",
                           "Type4-GAM-training.npy",
                           "Type5-GAM-training.npy"]

        train_x, valid_x = [], []
        train_y, valid_y = [], []
        a = {0:[],1:[],2:[],3:[],4:[]}
        for idx, file in enumerate(train_data_list):
            data = np.load(os.path.join(self.path, file))
            # assert np.array([ np.isnan(i).astype(int).sum()!=0 for i in data.tolist()]).astype(int).sum() == 0
            data_tr, data_va = train_test_split(data.tolist(), test_size=0.2, random_state=1126)

            print("\tThere are {} ({}+{}) log files in {} data with label {}.".format(len(data_tr)+len(data_va), len(data_tr), len(data_va), file, idx))
            tmp_x = []

            for file in data_tr:
                for row in file:
                    tmp_x.append(row)

            if idx == 4:
                tmp_x = random.sample(tmp_x, 50000)
            tmp_y = [idx]*len(tmp_x)

            train_x.extend(tmp_x)
            train_y.extend(tmp_y)
            print("\t\tThere are {} train data.".format(len(tmp_x)))

            tmp_x = []
            for file in data_va:
                for row in file:
                    tmp_x.append(row)

            if idx == 4:
                tmp_x = random.sample(tmp_x, 18000)
            tmp_y = [idx]*len(tmp_x)
            valid_x.extend(tmp_x)
            valid_y.extend(tmp_y)
            print("\t\tThere are {} valid data.".format(len(tmp_x)))

        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.valid_x = np.array(valid_x)
        self.valid_y = np.array(valid_y)
        print(self.train_x.shape)
        for i in range(3):
            new_train_x = self.random_rotate(np.array(train_x))
            self.train_x = np.concatenate((self.train_x,new_train_x),axis=0)
            self.train_y = np.concatenate((self.train_y,np.array(train_y)),axis=0)
        self.train_x =self.train_x.reshape(-1,128*9)
        print(self.train_x.shape)
        print(self.train_y.shape)
        self.train_num_x = len(self.train_x)
        self.valid_num_x = len(self.valid_x)
        print("There are {} training data.".format(self.train_num_x))
        print("There are {} validation data.".format(self.valid_num_x))

    def load_testdata(self):
        print("Load test data...")
        test_data_list = [
                "Type1-GAM-testing.npy",
                "Type2-GAM-testing.npy",
                "Type3-GAM-testing.npy",
                "Type4-GAM-testing.npy",
                "Type5-GAM-testing.npy"]
        test_x,test_y = [],[]
        data_test_by_file = []

        for idx, testfile in enumerate(test_data_list):
            test = np.load(os.path.join(self.path, testfile))
            # assert np.array([ np.isnan(i).astype(int).sum()!=0 for i in data.tolist()]).astype(int).sum() == 0
            print("\tThere are {} log files in {} data with label {}.".format(len(test), testfile, idx))
            for file in test:
                data_test_by_file.append((file,idx))
                for row in file:
                    flat_row = row.reshape(-1)
                    test_x.append(flat_row.tolist())
                    test_y.append(idx)
        self.test_x = np.array(test_x)
        self.test_y = np.array(test_y)
        self.test_data_by_file = data_test_by_file
        self.test_num_data = len(self.test_x)
        print("There are {} testing data".format(self.test_num_data))

    def create_model(self):
        print("Create model.")
        self.model = XGBClassifier() 

    def random_rotate(self,x):
        trans = random.sample(self.Trans,1)[0]

        xg = x[:,:,:3]
        xa = x[:,:,3:6]
        xm = x[:,:,6:]

        n_xg = np.dot(xg,trans)
        n_xa = np.dot(xa,trans)
        n_xm = np.dot(xm,trans)

        out = np.concatenate((n_xg,n_xa,n_xm),axis=2)
        return out

    def trainInit(self):
        self.load_data()
        self.create_model()

    def Add_feature(self, input_data): # input_data.shape: ( N, 130, 9 )
        # embed()
        x = np.array(input_data)

        bs = x.shape[0]
        x0 = x[:, :-2]
        x1 = x[:, 1:-1]
        x2 = x[:, 2:]

        diff0 = x1 - x0
        diff1 = x2 - x1
        ddiff0 = diff1 - diff0

        magG = ((x0[:, :, 0]**2 + x0[:, :, 1]**2 + x0[:, :, 2]**2) ** 0.5).reshape(-1, 128, 1)
        magA = ((x0[:, :, 3]**2 + x0[:, :, 4]**2 + x0[:, :, 5]**2) ** 0.5).reshape(-1, 128, 1)
        magM = ((x0[:, :, 6]**2 + x0[:, :, 7]**2 + x0[:, :, 8]**2) ** 0.5).reshape(-1, 128, 1)

        return np.concatenate((x0, diff0, ddiff0, magG, magA, magM), axis=2)
        # return np.array(input_data)[:, :128, ]

    def train(self):
        t1 = time.time()
        self.model.fit(self.train_x,self.train_y)

    def save_model(self,chkptfile):
        with open(chkptfile, 'wb') as wf:
            pickle.dump(final.model, wf, protocol=pickle.HIGHEST_PROTOCOL)


    def load_model(self, chkptfile):
        with open(chkptfile, 'rb') as rf:
            self.model = pickle.load(rf)

    def test(self):
        self.load_testdata()
        print("------------------{}-------------------".format("Test By Each 128 "))
        pred = self.model.predict(self.test_x)
        test_acc, label_list, pred_list, test_c_acc, test_len = [], [], [], [], 0

        acc = (pred==self.test_y)
        test_acc.extend(acc)
        print("test_acc {:.2f}%".format(np.mean(test_acc)*100))
        for c in range(5):
            label_arr = np.array(self.test_y)
            c_loc = np.nonzero(label_arr == c) 
            test_c_acc.append(np.mean(np.equal(pred[c_loc], c)))
            print("test_type {} acc {:.2f}%".format(c, test_c_acc[-1]*100))
        return np.mean(test_acc), test_c_acc

    def test_by_file(self):
        #self.load_testdata()
        print("------------------{}-------------------".format("Test By Each File "))
        f_acc = [[] for i in range(5)]
        for file, idx in self.test_data_by_file:
            file_test_x = []
            for row in file:
                file_test_x.append(row.reshape(-1))
            file_test_y = np.array([idx]*len(file_test_x))
            file_test_x = np.array(file_test_x)
            file_pred = []
            pred = self.model.predict(file_test_x)
            acc = np.mean(np.equal(pred,file_test_y))
            f_acc[idx].append(acc)
            # print(file_pred)
            # print(idx)
            # input()
        t = []
        for c in range(5):
            t.extend(f_acc[c])
        print("test_acc {:.2f}%".format(np.mean(t)*100))
        for c in range(5):
            print("test_type {} acc {:.2f}%".format(c, np.mean(f_acc[c])*100))
        print("-------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='data')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--lr_rate', type=float, default=1e-4)
    parser.add_argument('-md', '--model_dump', type=str, default='chkpt/xgb_aug.pickle')
    parser.add_argument('-o', '--output_csv', type=str, default='output.csv')
    parser.add_argument('-p', '--print_iter', type=int, default=1e3, help='Print every p iterations')
    parser.add_argument('-s', '--save_iter', type=int, default=30, help='Save every p iterations')
    args = parser.parse_args()

    final = main(args)
    final.create_model()
    final.load_data_aug()
    final.train()
    final.save_model(args.model_dump)
    final.load_model(args.model_dump)
    final.test()
    final.test_by_file()
    
    sys.exit()


    """
    final.load_data()
    t1 = time.time()
    gen = final.data_generator(64)
    for idx, (input_data, labels) in enumerate(gen):
        pass
    print("Spends {:2.f} seconds for generating train data with one epoch.".format(time.time() - t1))
    """
