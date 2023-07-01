import argparse
import sqlite3
import json
import os
import random
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from anyasand.model import AnyaAE
from anyasand.dictionary import DictionaryTrainer

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)
logger = logging.getLogger("anyasand-trainer")


class AnyaDatasets(Dataset):
    def __init__(self, corpus_data, dictionary, max_data_size=1000):
        self._corpus_data = corpus_data
        self._dict = dictionary
        datasize = max_data_size
        if len(self._corpus_data) < datasize:
            datasize = len(self._corpus_data)

        logging.info("Data read size= %d (SQL size=%d)" % (datasize, len(self._corpus_data)))
        data_idx_list = random.sample(list(range(len(self._corpus_data))), datasize)

        self._inv = []
        self._anv = []
        for data in data_idx_list:
            dat = json.loads(self._corpus_data[data][0])
            for i, word in enumerate(dat["words"]):
                if i == 0:
                    continue
                elif i == 1:
                    self._inv.append(self._dict.get_dwords(self._dict.wid_bos, dat["words"][i-1]))
                else:
                    self._inv.append(self._dict.get_dwords(dat["words"][i-2], dat["words"][i-1]))

                self._anv.append(self._dict.get_dwords(dat["words"][i-1], dat["words"][i]))

    def __len__(self):
        return len(self._inv)

    def __getitem__(self, idx):
        return self._inv[idx], self._anv[idx]


class Trainer:
    def __init__(self, dataset_path, db_path, epoch, device="cuda"):
        self._epoch = epoch
        self._device = device
        self._dict = DictionaryTrainer(db_path)
        self._criterion_vec = nn.MSELoss(reduction='sum')
        self._criterion_pos = nn.CrossEntropyLoss(reduction='sum')

        conn = sqlite3.connect(dataset_path)
        cur = conn.cursor()
        cur.execute("SELECT data FROM corpus")
        self._corpus_datas = cur.fetchall()
        cur.close()
        conn.close()

    def __call__(self, out_model_path, inner_loop=100000):
        model = AnyaAE(self._dict.input_vec_size).to(self._device)
        #if os.path.isfile(out_model_path):
        #    model.load_state_dict(torch.load(out_model_path))

        model = model.to(self._device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        for i in range(self._epoch):
            model = model.to(self._device)
            model.train()
            trn_data = AnyaDatasets(self._corpus_datas, self._dict, inner_loop)
            train_loader = torch.utils.data.DataLoader(trn_data, shuffle=True, pin_memory=True, batch_size=100)
            train_data_size = 0

            train_loss = 0
            for crt_t, next_t in tqdm(train_loader):
                x = crt_t.to(self._device)
                next_t = next_t.to(self._device)
                y = model(x)
                loss = self._compute_loss(y, next_t)

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_data_size += x.shape[1]

            model.eval()
            tst_data = AnyaDatasets(self._corpus_datas, self._dict, 2000)
            test_loader = torch.utils.data.DataLoader(tst_data, shuffle=True, pin_memory=True, batch_size=100)
            test_loss = 0.
            test_data_size = 0
            test_cnt = 0
            test_crr = 0
            with torch.no_grad():
                for crt_t, next_t in tqdm(test_loader):
                    x = crt_t.to(self._device)
                    next_t = next_t.to(self._device)
                    y = model(x)

                    # Compute loss
                    loss = self._compute_loss(y, next_t)
                    test_loss += loss.item()
                    test_data_size += x.shape[1]

                    # Compute acc
                    cnt, crr = self._compute_acc(y, next_t)
                    test_cnt += cnt
                    test_crr += crr

                logging.info("loss [%03d]: (train)%.8f  (test)%.8f, %.1f%%" %
                             ((i+1),
                              train_loss / train_data_size,
                              test_loss / test_data_size,
                              test_crr / test_cnt * 100))

            torch.save(model.to('cpu').state_dict(), out_model_path)

        # save
        self._dict.close()

        dummy_input = torch.randn((self._dict.input_vec_size,))
        torch.onnx.export(model, dummy_input, "anya.onnx", verbose=True)

    def _compute_loss(self, y, y_in):
        #loss_vec = self._criterion_vec(y[:8], y_in[:8])
        #loss_pos = self._criterion_pos(y[-self._dict.pos_len:], y_in[-self._dict.pos_len:])
        #return loss_vec + loss_pos
        return self._criterion_vec(y, y_in)

    def _compute_acc(self, y, y_in):
        cnt = 0
        correct = 0
        for i, y_res in enumerate(y):
            if y_res.dim() > 0:
                cnt += 1
                y_res_pos = torch.argmax(y_res[-self._dict.pos_len:])
                y_in_pos = torch.argmax(y_in[i][-self._dict.pos_len:])
                if y_res_pos.item() == y_in_pos.item():
                    correct += 1
        return cnt, correct


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--in_db_path', help='input training corpus path', default="./anya-corpus.db")
    arg_parser.add_argument('-d', '--db_path', help='dictionary database path', default="./anya-dic.db")
    arg_parser.add_argument('-o', '--out_path', help='output model file path', default="anya.mdl")
    arg_parser.add_argument('-e', '--epoch_num', help='epoch num', default=10)
    args = arg_parser.parse_args()

    trainer = Trainer(args.in_db_path, args.db_path, args.epoch_num)
    trainer(args.out_path)


if __name__ == "__main__":
    main()
