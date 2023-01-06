import argparse
import torch
import torch.nn as nn
import numpy as np
from anyasand.model import AnyaAE
from anyasand.dictionary import DictionaryConverter as Dictionary


class Converter:
    def __init__(self, dnn_model, db_path):
        self._dict = Dictionary(db_path)
        self._model = AnyaAE(self._dict.word_vec_size, self._dict.pos_len)
        self._model.load_state_dict(torch.load(dnn_model))
        self._criterion = nn.MSELoss(reduction='sum')

    def __call__(self, text):
        in_vec = np.array([[self._dict.get(self._dict.wid_bos)]], dtype=np.float32)
        y = self._model(torch.from_numpy(in_vec))

        word_tree = self._dict.build_word_tree(text)
        for i, ym_ary in enumerate(word_tree):
            print(ym_ary)
            min_score = 1.
            min_score_id = None
            for ym in ym_ary:
                words = self._dict.gets(ym)
                for wid in words:
                    score = self._score(in_vec, wid, y)
                    """
                    score = self._criterion(y, torch.from_numpy(self._dic.vector(word)).unsqueeze(0).unsqueeze(0))
                    print(" %s: %f" % (word.decode().split("_")[0], score))
                    if min_score > score:
                        min_score = score
                        min_score_id = word
                    """

                #print("  => %s" % min_score_id.decode().split("_")[0])

                #idx, score = self._score(vec, cost)
                #print(surfaces[idx], score)
            break

    def _score(self, in_vec, wid, y):
        get_vec = self._dict.get(wid.decode())
        in_vec_tmp = torch.from_numpy(get_vec)
        #in_vec_tmp = np.vstack((in_vec, [[get_vec]]))
        #y_tmp = self._model(torch.from_numpy(in_vec_tmp))
        y_tmp = self._model(torch.from_numpy(in_vec))
        score = self._criterion(y_tmp[0][0], in_vec_tmp)
        diff = y_tmp[0][0] - in_vec_tmp
        print(self._dict.wid2name(wid), score, diff)
        return score


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model_path', help='dnn model path', default="./anya.mdl")
    arg_parser.add_argument('-d', '--db_path', help='dictionary DB path', default="./anya-dic.db")
    arg_parser.add_argument('-t', '--text', help='convert text', required=True)
    args = arg_parser.parse_args()

    converter = Converter(args.model_path, args.db_path)
    converter(args.text)


if __name__ == "__main__":
    main()
