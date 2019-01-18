import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import time
import random
import argparse
import numpy as np

from utils import models as models
from sys import stdout
from IPython import embed
from scipy.fftpack import fft
from collections import Counter
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available else "cpu"

Draw = []

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
        train_data_list = [
                           "Type1-GAM-training.npy",
                           "Type2-GAM-training.npy",
                           "Type3-GAM-training.npy",
                           "Type4-GAM-training.npy",
                           "Type5-GAM-training.npy"]

        train_data, valid_data = [], []
        mt, mv = 0, 0
        for idx, file in enumerate(train_data_list):
            data = np.load(os.path.join(self.path, file))

            # assert np.array([ np.isnan(i).astype(int).sum()!=0 for i in data.tolist()]).astype(int).sum() == 0
            data_tr, data_va = train_test_split(data.tolist(), test_size=0.2, random_state=1126)

            print("\tThere are {} ({}+{}) log files in {} data with label {}.".format(len(data_tr)+len(data_va), len(data_tr), len(data_va), file, idx))
            tmp = []
            for file in data_tr:
                for row in file:
                    tmp.append((row, idx))
            if idx == 4:
                tmp = random.sample(tmp, mt)
            else:
                mt = max(len(tmp), mt)
            train_data.extend(tmp)
            print("\t\tThere are {} train data.".format(len(tmp)))
            tmp = []
            for file in data_va:
                for row in file:
                    tmp.append((row, idx))
            if idx == 4:
                tmp = random.sample(tmp, mv)
            else:
                mv = max(len(tmp), mv)
            valid_data.extend(tmp)
            print("\t\tThere are {} valid data.".format(len(tmp)))
            # print(mt, mv)

        self.train_data = train_data
        self.valid_data = valid_data
        self.train_num_data = len(self.train_data)
        self.valid_num_data = len(self.valid_data)
        print("There are {} training data.".format(self.train_num_data))
        print("There are {} validation data.".format(self.valid_num_data))

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


    
    def load_testdata(self):
        print("Load test data...")
        test_data_list = [
                "Type1-GAM-testing.npy",
                "Type2-GAM-testing.npy",
                "Type3-GAM-testing.npy",
                "Type4-GAM-testing.npy",
                "Type5-GAM-testing.npy"]
        data_te, data_te_by_file = [], []
        for idx, testfile in enumerate(test_data_list):
            test = np.load(os.path.join(self.path, testfile))
            # assert np.array([ np.isnan(i).astype(int).sum()!=0 for i in data.tolist()]).astype(int).sum() == 0
            print("\tThere are {} log files in {} data with label {}.".format(len(test), testfile, idx))
            for file in test:
                data_te_by_file.append((file, idx))
                for row in file:
                    data_te.append((row, idx))
        self.test_data = data_te
        self.test_num_data = len(self.test_data)
        self.test_data_by_file = data_te_by_file
        print("There are {} testing data".format(self.test_num_data))

    def create_model(self):
        print("Create model.")
        # self.model = models.RNNbase()
        self.model = models.RNNatt(input_size=39)
        print(self.model)
        self.model.to(device)

    def trainInit(self):
        self.load_data()
        self.create_model()
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-3)
        self.objective = nn.NLLLoss()


    def random_rotate(self, x):
        trans = random.sample(self.Trans, 1)[0]

        xg = x[:, :, :3]
        xa = x[:, :, 3:6]
        xm = x[:, :, 6:]

        n_xg = np.dot(xg, trans)
        n_xa = np.dot(xa, trans)
        n_xm = np.dot(xm, trans)

        out = np.concatenate((n_xg, n_xa, n_xm), axis=2)
        return out

    def x_y_z_rotate(self, x):
        rd = random.sample(range(3), 3)
        new = rd+([i+3 for i in rd])+([i+6 for i in rd])
        return x[:, :, new]

    def Add_feature(self, input_data, axis_aug=False): # input_data.shape: ( N, 130, 9 )
        x = np.array(input_data)
        if axis_aug:
            # x = self.x_y_z_rotate(x)
            x = self.random_rotate(x)
        

        bs = x.shape[0]
        x0 = x[:, :-2]
        x1 = x[:, 1:-1]
        x2 = x[:, 2:]

        diff0 = x1 - x0
        diff1 = x2 - x1
        ddiff0 = diff1 - diff0

        magG = ((x0[:, :, 0]**2 + x0[:, :, 1]**2 + x0[:, :, 2]**2) ** 0.5).reshape(bs, -1, 1)
        magA = ((x0[:, :, 3]**2 + x0[:, :, 4]**2 + x0[:, :, 5]**2) ** 0.5).reshape(bs, -1, 1)
        magM = ((x0[:, :, 6]**2 + x0[:, :, 7]**2 + x0[:, :, 8]**2) ** 0.5).reshape(bs, -1, 1)

        ver1 = np.concatenate((x0, diff0, ddiff0, magG, magA, magM), axis=2)
        ver2 = ver1.copy()
        # ver3 = np.concatenate((x0[:, :, 3:6], diff0[:, :, 3:6], ddiff0[:, :, 3:6], magA), axis=2)
        for i in range(9):
            y = x0[:, :, i]
            yr = fft(y).real.reshape(bs, -1, 1)
            ver2 = np.concatenate((ver2, yr), axis=2)
        # for i in range(3, 6):
        #     y = x0[:, :, i]
        #     yr = fft(y).real.reshape(bs, -1, 1)
        #     ver3 = np.concatenate((ver3, yr), axis=2)

        return ver2
        # return ver1
        # return x0

    def data_generator(self, data, batch_size, axis_aug=False, shuffle=True):
        # print(axis_aug)

        if shuffle:
            used_data = random.sample(data, len(data))
        else:
            used_data = np.copy(data)

        num_data = len(used_data)

        steps = num_data // batch_size if (num_data%batch_size)==0 else (num_data // batch_size) +1

        for i in range(steps):
            start = i * batch_size
            end = (i + 1) * batch_size 

            batch_data = used_data[start:end]
            input_data, labels = zip(*batch_data)

            input_data = self.Add_feature(input_data, axis_aug=axis_aug)
            input_data = torch.tensor(input_data, dtype=torch.float)

            labels = torch.tensor(labels, dtype=torch.long)
            yield input_data.to(device), labels.to(device)       


    def validation_acc(self, batch_size=64):

        val_gen = self.data_generator(self.valid_data, self.batch_size, shuffle=True)  # generate valid data
        self.model.eval()
        val_acc, val_loss, label_list, pred_list, val_c_acc, val_len = [], [], [], [], [], 0

        for (val_data, labels) in val_gen:
            val_len += len(val_data) 
            label_list.extend(labels.cpu().tolist())

            pred = self.model(val_data)
            loss = self.objective(pred, labels).data.cpu().item()

            pred_list.extend(pred.argmax(dim=1).cpu().tolist())
            acc = (pred.argmax(dim=1) == labels).float().cpu().tolist() 
            
            val_acc.extend(acc)
            val_loss.append(loss)

        # print(val_len)
        print("val_acc {:.2f}%".format(np.mean(val_acc)*100))
        for c in range(5):
            label_arr = np.array(label_list)
            pred_arr = np.array(pred_list)
            c_loc = np.nonzero(label_arr == c) 
            val_c_acc.append(np.mean(np.equal(pred_arr[c_loc], c)))
            print("val_type {} acc {:.2f}%".format(c, val_c_acc[-1]*100))

        return np.mean(val_acc)




    def train(self, epoch):
        steps = self.train_num_data // self.batch_size
        gen = self.data_generator(self.train_data, self.batch_size, axis_aug=True ,shuffle=True)  # generate train data

        t1 = time.time()
        epoch_loss = []
        epoch_acc = []
        self.model.train(True)

        for idx, (input_data, labels) in enumerate(gen):
            # Forward and backward.
            self.optim.zero_grad()

            pred = self.model(input_data)
            loss = self.objective(pred, labels)

            loss.backward()
            self.optim.step()

            # clip = 50.0
            # _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)

            loss = loss.data.cpu().item()
            acc = (pred.argmax(dim=1) == labels).float().mean().data.item()
            epoch_loss.append(loss)
            epoch_acc.append(acc)

            Iter = 100.0 * (idx + 1) / steps
            stdout.write("\rEpoch: {}/{}, Iter: {:.1f}%, Loss: {:.4f}, Acc: {:.4f}".format(epoch, self.epochs, Iter, np.mean(epoch_loss), np.mean(epoch_acc)))
            if (idx + 1) % self.print_iter == 0 :
                print(" ")

        print(" Spends {:.2f} seconds.".format(time.time() - t1))
        print("The Training dataset Accuracy is {:.2f}%, Loss is {:.4f}".format(np.mean(epoch_acc)*100, np.mean(epoch_loss)))

        D_tr = np.mean(epoch_acc)
        D_va = self.validation_acc()
        D_te = self.test()

        acc_valid = D_va

        self.loss_epoch_tr.append(np.mean(epoch_loss))
        self.acc_epoch_tr.append(np.mean(epoch_acc))
        
        if epoch ==1:
            self.save_model()
        elif acc_valid > np.max(self.acc_epoch_va):
            self.save_model()

        self.acc_epoch_va.append(acc_valid)
        # self.loss_epoch_va.append(loss_valid)
        Draw.append((D_tr, D_va, D_te))

    def save_model(self):
        torch.save({
            'epoch': self.epochs,
            'model': self.model.state_dict(),
            'opt': self.optim.state_dict(),
            'loss_epoch_tr': self.loss_epoch_tr,
            'acc_epoch_tr': self.acc_epoch_tr,
            'loss_epoch_va': self.loss_epoch_va,
            'acc_epoch_va': self.acc_epoch_va,
        }, self.model_dump)


    def loadchkpt(self, ckptfile):
        ckpt = torch.load(ckptfile)
        self.model.load_state_dict(ckpt['model'])

    def trainIter(self):
        self.trainInit()
        self.load_testdata()
        for epoch in range(self.epochs):
            self.train(epoch + 1)

    def testInit(self, args):
        self.load_testdata()
        self.create_model()
        self.loadchkpt(args.model_dump)
        self.test()


    def test(self):
        test_gen = self.data_generator(self.test_data, self.batch_size, shuffle=False)  #generate test data
        self.model.eval()

        self.test_by_file(self.test_data_by_file)

        test_acc, label_list, pred_list, test_c_acc, test_len = [], [], [], [], 0

        for (test_data,labels) in test_gen:
            test_len += len(test_data) 
            label_list.extend(labels.cpu().tolist())
            pred = self.model(test_data)
            pred_list.extend(pred.argmax(dim=1).cpu().tolist())
            acc = (pred.argmax(dim=1)==labels).float().cpu().tolist() 
            test_acc.extend(acc)
        print(test_len)
        print("test_acc {:.2f}%".format(np.mean(test_acc)*100))
        for c in range(5):
            label_arr = np.array(label_list)
            pred_arr = np.array(pred_list)
            c_loc = np.nonzero(label_arr == c) 
            test_c_acc.append(np.mean(np.equal(pred_arr[c_loc], c)))
            print("test_type {} acc {:.2f}%".format(c, test_c_acc[-1]*100))
        return np.mean(test_acc)#, test_c_acc

    def test_by_file(self, data_by_file):
        print("------------------{}-------------------".format("Test By Each File "))
        f_acc = [[] for i in range(5)]
        for file, idx in data_by_file:
            tmp_data = [(i, idx)for i in file]
            tmp_gen = self.data_generator(tmp_data, self.batch_size, shuffle=True)  # generate valid data
            file_pred = []
            for (d, l) in tmp_gen:
                pred = self.model(d)
                file_pred.extend(pred.argmax(dim=1).cpu().tolist())
            c = Counter(file_pred)
            out = c.most_common(1)[0][0]
            f_acc[idx].append(int(out == idx))
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
    parser.add_argument('-dp', '--data_path', type=str, default='./data')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--lr_rate', type=float, default=1e-4)
    parser.add_argument('-md', '--model_dump', type=str, default='./chkpt/model.tar')
    parser.add_argument('-o', '--output_csv', type=str, default='output.csv')
    parser.add_argument('-p', '--print_iter', type=int, default=1e3, help='Print every p iterations')
    parser.add_argument('-s', '--save_iter', type=int, default=30, help='Save every p iterations')
    parser.add_argument('-tr', '--train', type=int, default=1, help='Train and test: 1, Only test: 0')

    args = parser.parse_args()

    final = main(args)
    if args.train:
        final.trainIter()
    np.save("Draw.npy", Draw)
    final.testInit(args)

    sys.exit()


    """
    final.load_data()
    t1 = time.time()
    gen = final.data_generator(64)
    for idx, (input_data, labels) in enumerate(gen):
        pass
    print("Spends {:2.f} seconds for generating train data with one epoch.".format(time.time() - t1))
    """
