import argparse
import torch
import torch.nn as nn
import numpy as np
from anyasand.model import AnyaAE
from anyasand.dictionary import DictionaryConverter as Dictionary


class Converter:
    def __init__(self, dnn_model, db_path):
        self._dict = Dictionary(db_path)
        self._model = AnyaAE(self._dict.input_vec_size)
        self._model.load_state_dict(torch.load(dnn_model))
        self._criterion = nn.MSELoss(reduction='sum')

    def __call__(self, text):
        fixed = []
        word_tree = self._dict.build_word_tree(text)
        print(word_tree)

        for crt_idx, ym_ary in enumerate(word_tree):
            print(crt_idx, " -> ")
            min_words = {"words": [], "cost": 0.}

            for ym in ym_ary:
                pre_idx = crt_idx - len(ym)
                if pre_idx >= 0:
                    for pre_ym in word_tree[pre_idx]:
                        if pre_idx - len(pre_ym) < 0:
                            for pre_word in self._dict.gets(pre_ym):
                                words = [pre_word.decode()]
                                in_vec = self._get_in_vec(words)
                                pre_word_vec = self._model(in_vec)[0]
                                predict_word_vec = self._model(in_vec)[1]
                                vec = torch.from_numpy(self._dict.get(pre_word.decode()))
                                pre_score = self._criterion(pre_word_vec, vec)
                                pre_score = pre_score.item()

                                for next_word in self._dict.gets(ym):
                                    vec = torch.from_numpy(self._dict.get(next_word.decode()))
                                    score = self._criterion(predict_word_vec, vec)
                                    score = pre_score + score.item()

                                    copied = words.copy()
                                    copied.append(next_word.decode())

                                    if min_words["cost"] == 0. or score < min_words["cost"]:
                                        min_words["words"] = copied
                                        min_words["cost"] = score

                                        disp_words = []
                                        for i, wid in enumerate(copied):
                                            disp_words.append(self._dict.wid2name(wid))
                                        print(" ",
                                              disp_words,
                                              min_words["cost"])

                        else:
                            words = fixed[pre_idx]["words"].copy()
                            in_vec = self._get_in_vec(words)
                            predict_word_vec = self._model(in_vec)[-1:]

                            for next_word in self._dict.gets(ym):
                                vec = torch.from_numpy(self._dict.get(next_word.decode()))
                                score = self._criterion(predict_word_vec, vec)
                                score = score.item() + fixed[pre_idx]["cost"]

                                copied = words.copy()
                                copied.append(next_word.decode())

                                if min_words["cost"] == 0. or score < min_words["cost"]:
                                    min_words["words"] = copied
                                    min_words["cost"] = score

                                    disp_words = []
                                    for i, wid in enumerate(copied):
                                        disp_words.append(self._dict.wid2name(wid))
                                    print(" ",
                                          disp_words,
                                          min_words["cost"])

            fixed.append(min_words)

    def _get_in_vec(self, words):
        in_vec = self._dict.get(self._dict.wid_bos)
        for wid in words:
            in_vec = np.vstack((in_vec, self._dict.get(wid)))
        return torch.from_numpy(in_vec)


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
