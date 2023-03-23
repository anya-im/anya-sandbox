import sqlite3
import json
import argparse
from tqdm import tqdm


class CostCalculator:
    def __init__(self, dic_db_path="./anya-dic.db", corpus_db_path="./anya-corpus.db"):
        self._dic_db_path = dic_db_path
        self._corpus_db_path = corpus_db_path
        self._read = {}
        self._word_cnt = {}
        self._read_cnt = {}

        conn = sqlite3.connect(dic_db_path)
        cur = conn.cursor()
        cur.execute('SELECT id, read FROM words;')
        for data in cur.fetchall():
            self._read[data[0]] = data[1]
            self._word_cnt[data[0]] = 0
            self._read_cnt[data[1]] = 1

    def __call__(self):
        self._parse_corpus()
        self._record_cost()

    def _parse_corpus(self):
        conn = sqlite3.connect(self._corpus_db_path)
        cur = conn.cursor()
        cur.execute("SELECT data FROM corpus")
        corpus = cur.fetchall()

        for data in tqdm(corpus):
            words = json.loads(data[0])["words"]
            for wid in words:
                self._word_cnt[wid] += 1
                self._read_cnt[self._read[wid]] += 1

        cur.close()
        conn.close()

    def _record_cost(self):
        conn = sqlite3.connect(self._dic_db_path)
        cur = conn.cursor()

        cur.execute("select count(*) from sqlite_master where type='table' and name='cost'")
        if cur.fetchone()[0] > 0:
            cur.execute("DROP TABLE cost;")
        cur.execute("CREATE TABLE cost(id INTEGER, val REAL);")

        for wid, read in self._read.items():
            cost = 1 - (self._word_cnt[wid] / self._read_cnt[read])
            cur.execute("INSERT INTO cost(id, val) values(?, ?);", (wid, cost))

        cur.execute('CREATE INDEX cost_idx ON cost(id);')
        cur.execute('commit;')
        cur.close()
        conn.close()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--dic_db_path', help='dictionary DB path', default="./anya-dic.db")
    arg_parser.add_argument('-c', '--corpus_db_path', help='corpus DB path', default="./anya-corpus.db")
    args = arg_parser.parse_args()
    calculator = CostCalculator(args.dic_db_path, args.corpus_db_path)
    calculator()


if __name__ == "__main__":
    main()

