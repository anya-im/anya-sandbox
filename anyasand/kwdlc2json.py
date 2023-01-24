import json
import os
import argparse
import glob
from tqdm import tqdm
from anyasand.dictionary import DictionaryTrainer


class Parser:
    def __init__(self, in_path, out_path, dic_db_path="./anya-dic.db", wakati_txt="kwdlc_wakati.txt"):
        self._in_path = in_path
        self._dictionary = DictionaryTrainer(dic_db_path)
        abs_out_path = os.path.abspath(out_path)
        self._out_path_train = abs_out_path + "/train/"
        self._out_path_test = abs_out_path + "/test/"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        self._out_wakati = None
        if wakati_txt is not None:
            self._out_wakati = abs_out_path + "/" + wakati_txt

    def __call__(self):
        wakati_fp = None
        if self._out_wakati is not None:
            wakati_fp = open(self._out_wakati, "w", encoding='UTF-8')

        if not os.path.isdir(self._out_path_train):
            os.makedirs(self._out_path_train)
        self._parse_files(self._in_path + "/id/train.id", self._out_path_train, wakati_fp)

        if not os.path.isdir(self._out_path_test):
            os.makedirs(self._out_path_test)
        self._parse_files(self._in_path + "/id/test.id", self._out_path_test, wakati_fp)

        self._dictionary.commit()
        self._dictionary.close()

        if wakati_fp is not None:
            wakati_fp.close()

    def _parse_files(self, id_file, out_path, wakati_fp):
        in_path_list = []
        with open(os.path.abspath(id_file)) as f:
            for file in f.readlines():
                read_str = file.rstrip('\n')
                in_path = os.path.abspath(self._in_path + "/knp/" + read_str[:-5] + "/" + read_str + ".knp")
                in_path_list.append(in_path)

            with tqdm(total=len(in_path_list)) as progress:
                for in_path in in_path_list:
                    for file in glob.glob(in_path):
                        self._parse(file, out_path, wakati_fp)
                    progress.update()

    def _parse(self, in_file, out_path, wakati_fp):
        with open(in_file) as f:
            bodies = []
            body = None

            for line in f.readlines():
                line_split = line.split()

                # 文
                if line_split[0] == "#":
                    body = {"id": line_split[1]}
                    chunks = []
                    chunk = None

                # 文節
                elif line_split[0] == "*":
                    if chunk is not None:
                        chunk["words"] = words
                        chunks.append(chunk)

                    chunk = {"link_idx": int(line_split[1][:-1])}
                    words = []

                # 補足？
                elif line_split[0] == "+":
                    pass

                # 文末
                elif line_split[0] == "EOS":
                    if body is not None:
                        body["chunks"] = chunks
                        bodies.append(body)
                    if chunk is not None:
                        chunk["words"] = words
                        chunks.append(chunk)

                    # 係り受け情報の追加
                    for chunk in chunks:
                        if chunk["link_idx"] > 0:
                            chunk["link"] = chunks[chunk["link_idx"]]["words"]

                # 単語
                else:
                    words.append(
                        self._dictionary.wid_insert(line_split[0], line_split[1], line_split[3], line_split[5])
                    )

        words_of_files = []
        for body in bodies:
            words = []
            for chunk in body["chunks"]:
                for word in chunk["words"]:
                    words.append(word)

                    if wakati_fp is not None:
                        name = self._dictionary.get_name(word)
                        wakati_fp.write(name + " ")

            words_of_files.append(words)
            if wakati_fp is not None:
                wakati_fp.write("\n")

        base, _ = os.path.splitext(os.path.basename(in_file))
        out_file = out_path + "/" + base

        to_json = {"bodies": words_of_files}
        with open(out_file + ".json", 'w') as of:
            json.dump(to_json, of, ensure_ascii=False)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--in_path', help='KWDLC path', required=True)
    arg_parser.add_argument('-o', '--out_path', help='output path', default="out/")
    args = arg_parser.parse_args()
    parser = Parser(args.in_path, args.out_path)
    parser()


if __name__ == "__main__":
    main()

