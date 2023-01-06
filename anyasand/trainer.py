import argparse
import glob
import json
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from anyasand.model import AnyaAE
from anyasand.dictionary import DictionaryTrainer


class AnyaDatasets(Dataset):
    def __init__(self, in_path, dictionary):
        self._dict = dictionary
        self._word_index = []

        for name in glob.glob(in_path + '/*.json'):
            with open(name, "r") as f:
                data = json.load(f)
                for body in data["bodies"]:
                    word_index = [[self._dict.wid_bos], [body[0]]]
                    for i, _ in enumerate(body):
                        if i + 1 < len(body):
                            word_index[0].append(body[i])
                            word_index[1].append(body[i+1])
                    self._word_index.append(word_index)

    def __len__(self):
        return len(self._word_index)

    def __getitem__(self, idx):
        crt_t = np.empty((0, self._dict.word_vec_size + self._dict.pos_len), dtype="float32")
        next_t = np.empty((0, self._dict.word_vec_size + self._dict.pos_len), dtype="float32")

        for i in range(len(self._word_index[idx][0])):
            crt_t_tmp, crt_eye_tmp = self._dict.get(self._word_index[idx][0][i])
            crt_t_tmp = np.concatenate([crt_t_tmp, crt_eye_tmp])
            crt_t = np.vstack((crt_t, crt_t_tmp))

            next_t_tmp, next_eye_tmp = self._dict.get(self._word_index[idx][1][i])
            next_t_tmp = np.concatenate([next_t_tmp, next_eye_tmp])
            next_t = np.vstack((next_t, next_t_tmp))

        return crt_t, next_t


class Trainer:
    def __init__(self, dataset_path, db_path, device="cuda"):
        self._device = device
        self._dict = DictionaryTrainer(db_path)
        self._trn_data = AnyaDatasets(dataset_path + "/train/", self._dict)
        self._tst_data = AnyaDatasets(dataset_path + "/test/", self._dict)
        self._criterion = nn.MSELoss(reduction='sum')

    def __call__(self, out_model_path, epoch):
        train_loader = torch.utils.data.DataLoader(self._trn_data, batch_size=1, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(self._tst_data, batch_size=1, shuffle=True, pin_memory=True)

        model = AnyaAE(self._dict.word_vec_size, self._dict.pos_len).to(self._device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

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
            cnt += 1
            y_res_pos = torch.argmax(y_res[self._dict.word_vec_size:])
            y_in_pos = torch.argmax(y_in[0][i][self._dict.word_vec_size:])
            if y_res_pos.item() == y_in_pos.item():
                correct += 1
        return cnt, correct


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--in_path', help='input training directory path', default="./out/")
    arg_parser.add_argument('-d', '--db_path', help='dictionary database path', default="./anya-dic.db")
    arg_parser.add_argument('-o', '--out_path', help='output model file path', default="anya.mdl")
    arg_parser.add_argument('-e', '--epoch_num', help='epoch num', default=10)
    args = arg_parser.parse_args()

    trainer = Trainer(args.in_path, args.db_path)
    trainer(args.out_path, args.epoch_num)


if __name__ == "__main__":
    main()
