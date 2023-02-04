import argparse
import sqlite3
import json
import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from anyasand.model import AnyaAE
from anyasand.dictionary import DictionaryTrainer


class AnyaDatasets(Dataset):
    def __init__(self, in_path, dictionary, data_size=1000):
        self._dict = dictionary
        self._bi_gram_vec = []

        conn = sqlite3.connect(in_path)
        cur = conn.cursor()

        cur.execute("SELECT data FROM corpus")
        sql_results = cur.fetchall()
        print("Data read size= %d (SQL data size=%d)" % (data_size, len(sql_results)))
        data_idx_list = random.sample(list(range(0, len(sql_results))), data_size)

        for idx in tqdm(data_idx_list):
            data = json.loads(sql_results[idx][0])
            for i, body in enumerate(data["words"]):
                if i == 0:
                    inv = self._dict.get_sword(self._dict.wid_bos)
                    anv = self._dict.get_sword(data["words"][0])
                elif i + 1 < len(data["words"]):
                    inv = np.vstack((inv, self._dict.get_sword(data["words"][i])))
                    anv = np.vstack((anv, self._dict.get_sword(data["words"][i + 1])))
                else:
                    pass
            self._bi_gram_vec.append([inv, anv])

        cur.close()
        conn.close()

    def __len__(self):
        return len(self._bi_gram_vec)

    def __getitem__(self, idx):
        return self._bi_gram_vec[idx][0], self._bi_gram_vec[idx][1]


class Trainer:
    def __init__(self, dataset_path, db_path, device="cuda"):
        self._device = device
        self._dict = DictionaryTrainer(db_path)
        self._trn_data = AnyaDatasets(dataset_path, self._dict, 200000)
        self._tst_data = AnyaDatasets(dataset_path, self._dict, 2000)
        self._criterion = nn.MSELoss(reduction='sum')

    def __call__(self, out_model_path, epoch):
        train_loader = torch.utils.data.DataLoader(self._trn_data, batch_size=1, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(self._tst_data, batch_size=1, shuffle=True, pin_memory=True)

        model = AnyaAE(self._dict.input_vec_size).to(self._device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        for i in range(epoch):
            model.train()
            train_data_size = 0

            train_loss = 0
            for crt_t, next_t in tqdm(train_loader):
                x = crt_t.to(self._device)
                next_t = next_t.to(self._device)
                y = model(x)
                loss = self._criterion(y, next_t)

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_data_size += x.shape[1]

            model.eval()
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
                    loss = self._criterion(y, next_t)
                    test_loss += loss.item()
                    test_data_size += x.shape[1]

                    # Compute acc
                    cnt, crr = self._compute_acc(y, next_t)
                    test_cnt += cnt
                    test_crr += crr

                print("loss [%03d]: (train)%.8f  (test)%.8f, %.1f%%" %
                      ((i+1),
                       train_loss / train_data_size,
                       test_loss / test_data_size,
                       test_crr / test_cnt * 100,
                       ))

        # save
        self._dict.close()
        torch.save(model.to('cpu').state_dict(), out_model_path)

    def _compute_acc(self, y, y_in):
        cnt = 0
        correct = 0
        for i, y_res in enumerate(y[0]):
            if y_res.dim() > 0:
                cnt += 1
                y_res_pos = torch.argmax(y_res[-self._dict.pos_len:])
                y_in_pos = torch.argmax(y_in[0][i][-self._dict.pos_len:])
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

    trainer = Trainer(args.in_db_path, args.db_path)
    trainer(args.out_path, args.epoch_num)


if __name__ == "__main__":
    main()
