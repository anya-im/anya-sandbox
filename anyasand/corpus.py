import json
import glob
import argparse
import logging
import sqlite3
from sudachipy import tokenizer
from sudachipy import dictionary

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)
logger = logging.getLogger("anyasand-corpus")


class Corpus:
    def __init__(self, in_dir="./corpus", corpus_db_path="./anya-corpus.db", dic_db_path="./anya-dic.db"):
        self._in_dir = in_dir
        self._corpus_db_path = corpus_db_path
        self._tokenizer = dictionary.Dictionary(dict="full").create()
        self._wakati_file = "./anya-corpus-wakati.txt"
        self._wakati_file_p = None

        self._pos_id = {}
        self._word_id = {}
        self._initialize_words(dic_db_path)

    def _initialize_words(self, dic_db_path):
        conn = sqlite3.connect(dic_db_path)
        cur = conn.cursor()

        cur.execute('SELECT id, name FROM positions;')
        for data in cur.fetchall():
            self._pos_id[data[1]] = data[0]

        cur.execute('SELECT id, name, pos FROM words;')
        for data in cur.fetchall():
            key = str(data[1]) + "__" + str(data[2])
            self._word_id[key] = data[0]

        cur.close()
        conn.close()

    def __call__(self):
        self._wakati_file_p = open(self._wakati_file, "w")

        conn = sqlite3.connect(self._corpus_db_path)
        cur = conn.cursor()
        cur.execute("select count(*) from sqlite_master where type='table' and name='corpus'")
        if cur.fetchone()[0] == 0:
            cur.execute("CREATE TABLE corpus(id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT);")

        exist_cnt = 0
        none_cnt = 0
        for file_name in glob.glob(self._in_dir + "/*.txt"):
            logging.info("PARSE start => %s" % file_name)
            with open(file_name) as f:
                for read_line in f:
                    read_line = read_line.replace('　', '')
                    read_line = read_line.replace('\n', '')

                    for text in read_line.replace("。", "。____").split("____"):
                        ret = self.parse(text, self._wakati_file_p)
                        if ret is not None:
                            cur.execute('INSERT INTO corpus(data) values(?);', (json.dumps(ret),))
                            exist_cnt += 1
                        else:
                            none_cnt += 1

        cur.execute('commit;')
        cur.close()
        conn.close()

        self._wakati_file_p.close()

        logging.info("ALL PARSE end ( %d / %d )" % (exist_cnt, none_cnt))

    def parse(self, text, wakati_file_p):
        ret = None
        words = []
        token = self._tokenizer.tokenize(text, tokenizer.Tokenizer.SplitMode.C)

        try:
            for m in token:
                surface = m.surface()
                pos_ary = m.part_of_speech()
                pos = pos_ary[0] + "." + pos_ary[1] + "." + pos_ary[2] + "." + pos_ary[3]
                key = surface + "__" + str(self._pos_id[pos])
                words.append(self._word_id[key])
                ret = {"text": text, "words": words}

        except KeyError:
            # print("KeyError: surface= %s pos_ary= %s" % (surface, str(pos_ary)))
            pass

        wakati = []
        for m in token:
            wakati.append(m.surface())
        wakati_file_p.write(" ".join(wakati) + " ")

        return ret


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--in_corpus', help='corpus path', default="./corpus")
    arg_parser.add_argument('-c', '--corpus_db', help='corpus DB path', default="./anya-corpus.db")
    arg_parser.add_argument('-d', '--dic_db', help='dictionary DB path', default="./anya-dic.db")
    args = arg_parser.parse_args()

    corpus = Corpus(args.in_corpus, args.corpus_db, args.dic_db)
    corpus()


if __name__ == "__main__":
    main()
