import argparse
import sqlite3
import json
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
    def __init__(self, in_path, dictionary, training_rate=0.2, max_data_size=1000):
        self._dict = dictionary
        self._training_rate = training_rate
        datasize = max_data_size

        conn = sqlite3.connect(in_path)
        cur = conn.cursor()

        cur.execute("SELECT data FROM corpus")
        self._sql_results = cur.fetchall()
        if len(self._sql_results) < datasize:
            datasize = len(self._sql_results)

        logging.info("Data read size= %d (SQL size=%d) rate=%.2f" % (datasize, len(self._sql_results), training_rate))
        self._data_idx_list = random.sample(list(range(len(self._sql_results))), datasize)

        cur.close()
        conn.close()

    def __len__(self):
        return len(self._data_idx_list)

    def __getitem__(self, idx):
        data = json.loads(self._sql_results[self._data_idx_list[idx]][0])
        data_len = len(data)
        teachers = random.sample(list(range(data_len)), int(data_len * (1. - self._training_rate)))
        for i, word in enumerate(data["words"]):
            if i in teachers:
                if i == 0:
                    inv = self._dict.get_sword(self._dict.wid_bos)
                    anv = self._dict.get_sword(data["words"][0])
                else:
                    inv = np.vstack((inv, self._dict.get_sword(data["words"][i - 1])))
                    anv = np.vstack((anv, self._dict.get_sword(data["words"][i])))
            else:
                if i == 0:
                    inv = self._dict.get_random_word(self._dict.wid_bos)
                    anv = self._dict.get_sword(data["words"][0])
                else:
                    inv = np.vstack((inv, self._dict.get_random_word(data["words"][i - 1])))
                    anv = np.vstack((anv, self._dict.get_sword(data["words"][i])))

        return (inv, anv), idx

    def update_vec(self, idx, vec):
        data = json.loads(self._sql_results[self._data_idx_list[idx]][0])
        vec_n = np.squeeze(vec.to('cpu').detach().numpy().copy())
        if vec_n.ndim > 1:
            for i, word in enumerate(vec_n):
                self._dict.set_new_word_vec(data["words"][i], word)


class Trainer:
    def __init__(self, dataset_path, db_path, device="cuda"):
        self._device = device
        self._dict = DictionaryTrainer(db_path)
        self._trn_data = AnyaDatasets(dataset_path, self._dict, 0.2, 200000)
        self._tst_data = AnyaDatasets(dataset_path, self._dict, 0., 2000)
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
            for (crt_t, next_t), idx in tqdm(train_loader):
                x = crt_t.to(self._device)
                next_t = next_t.to(self._device)
                y = model(x)
                loss = self._criterion(y, next_t)
                self._trn_data.update_vec(idx, y)

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
                for (crt_t, next_t), _ in tqdm(test_loader):
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

                logging.info("loss [%03d]: (train)%.8f  (test)%.8f, %.1f%%" %
                             ((i+1),
                              train_loss / train_data_size,
                              test_loss / test_data_size,
                              test_crr / test_cnt * 100))

        # save
        self._dict.commit_vec()
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
