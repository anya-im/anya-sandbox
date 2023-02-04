import csv
import os
import sqlite3
import logging
import argparse
import struct
import yaml
import jaconv
import marisa_trie
import numpy as np
from abc import ABCMeta


class Dictionary(metaclass=ABCMeta):
    def __init__(self, db_path="./anya-dic.db", vec_path=None, vec_size=8, initialize=False):
        self._words_vec = None
        self._vec_size = vec_size
        self._conn = sqlite3.connect(db_path)
        self._cur = self._conn.cursor()
        self._pos_len = 0

        if vec_path is not None:
            self._words_vec = self._read_vec(vec_path)

        if not initialize:
            self._cur.execute('SELECT count(*) FROM positions;')
            self._pos_len = self._cur.fetchone()[0]

            search_key = "_BOS"
            self._cur.execute('SELECT id FROM words WHERE name = ?;', (search_key,))
            self._wid_BOS = self._cur.fetchone()[0]
            self._is_update = False

            self._words, self._trie = self._build_db()

        self._pid_eye = np.eye(self.pos_len, dtype="float32")

    @property
    def wid_bos(self):
        return self._wid_BOS

    @property
    def single_vec_size(self):
        return self.word_vec_size + self.pos_len

    @property
    def input_vec_size(self):
        return self.single_vec_size

    def _vec_eye(self, wid):
        return self._pid_eye[(self._words[str(wid)]["pos"] - 1)]

    def _build_db(self):
        words = {}
        trie_key = []
        trie_val = []

        self._cur.execute('SELECT id, name, read, pos, vec FROM words;')
        for data in self._cur.fetchall():
            wid = str(data[0])
            words[wid] = {
                "name": data[1],
                "read": data[2],
                "pos": data[3],
                "vec": np.array(struct.unpack('=%df' % self._vec_size, data[4]), dtype="float32")
            }
            trie_key.append(data[2])
            trie_val.append(wid.encode())

        trie = marisa_trie.BytesTrie(zip(trie_key, trie_val))
        return words, trie

    def wid2name(self, wid):
        return self._words[wid]["name"]

    def wid2vec(self, wid):
        return self._words[wid.decode()]["vec"]

    @staticmethod
    def _read_vec(vec_path):
        vec_dic = {}
        with open(vec_path, encoding='utf8') as f:
            line_header = True
            for line in f:
                if not line_header:
                    row = line.split(" ")
                    vec = np.array([float(row[1]), float(row[2]), float(row[3]), float(row[4])], dtype=np.float16)
                    vec_dic[row[0]] = struct.pack('=4f', *vec)
                line_header = False
        return vec_dic

    def commit(self):
        if self._is_update:
            self._cur.execute('commit;')
        self._is_update = False

    def close(self):
        self._cur.close()
        self._conn.close()

    @property
    def vec_size(self):
        return self.word_vec_size + self.pos_len

    @property
    def word_vec_size(self):
        return self._vec_size

    @property
    def pos_len(self):
        return self._pos_len

    def _insert_pid(self, grp_name, sub_name):
        self._cur.execute('SELECT id FROM positions_group where name = ?;', (grp_name,))
        pid = self._cur.fetchone()
        if pid is None:
            self._cur.execute('INSERT INTO positions_group(name) values(?);', (grp_name,))
            self._cur.execute('SELECT id FROM positions_group where name = ?;', (grp_name,))
            pid = self._cur.fetchone()

        self._cur.execute('INSERT INTO positions(group_id, sub_name) values(?, ?);', (pid[0], sub_name))
        self._is_update = True

    def _pid_from_name(self, grp_name, sub_name):
        search_sql_str = "SELECT T1.id from positions as T1 join " \
                         "positions_group as T2 on T1.group_id = T2.id where T2.name=? and T1.sub_name=?;"
        self._cur.execute(search_sql_str, (grp_name, sub_name))
        res = self._cur.fetchone()
        if res is not None:
            pos_id = res[0]

        else:
            # Oops. add new record
            self._insert_pid(grp_name, sub_name)

            # research
            self._cur.execute(search_sql_str, (grp_name, sub_name))
            res = self._cur.fetchone()
            pos_id = res[0]

        return pos_id

    def _insert_word(self, name, read, pos_id):
        if self._words_vec is not None and name in self._words_vec:
            vec = self._words_vec[name]
        else:
            vec_np = np.random.rand(self._vec_size)
            vec = struct.pack('=%df' % vec_np.size, *vec_np)
        self._cur.execute('INSERT INTO words(name, read, pos, vec) values(?, ?, ?, ?);',
                          (name, read, pos_id, vec))
        self._is_update = True

    def get_name(self, wid):
        self._cur.execute('SELECT name FROM words where id = ?;', (wid,))
        return self._cur.fetchone()[0]


class DictionaryTrainer(Dictionary):
    def __init__(self, db_path="./anya-dic.db", vec_path="./anya-fasttext.vec"):
        super().__init__(db_path, vec_path)

    def get(self, wid):
        return self._words[str(wid)]["vec"], self._vec_eye(wid)

    def get_sword(self, wid):
        return np.concatenate([self._words[str(wid)]["vec"], self._vec_eye(wid)])

    def get_dwords(self, wid_f, wid_s):
        return np.concatenate([self._words[str(wid_f)]["vec"],
                               self._vec_eye(wid_f),
                               self._words[str(wid_s)]["vec"],
                               self._vec_eye(wid_s)])

    def wid_insert(self, name, read, grp_name, sub_name):
        pos_id = self._pid_from_name(grp_name, sub_name)
        self._cur.execute('SELECT id FROM words WHERE name = ? AND read = ? AND pos = ?;', (name, read, pos_id))
        res = self._cur.fetchone()
        if res is not None:
            # found. return word id
            ret_id = res[0]
        else:
            # Not found. insert to database
            self._insert_word(name, read, pos_id)

            # return word id
            self._cur.execute('SELECT id FROM words WHERE name = ? AND read = ? AND pos = ? LIMIT 1;', (name, read, pos_id))
            ret_id = self._cur.fetchone()[0]

        return ret_id


class DictionaryConverter(Dictionary):
    def __init__(self, db_path="./anya-dic.db", vec_path="./anya-fasttext.vec"):
        super().__init__(db_path, vec_path)

    def get(self, wid):
        return np.concatenate([self._words[str(wid)]["vec"], self._pid_eye[(self._words[str(wid)]["pos"] - 1)]])

    def get_dwords(self, wid_f, wid_s):
        return np.concatenate([self._words[str(wid_f)]["vec"],
                               self._vec_eye(wid_f),
                               self._words[str(wid_s)]["vec"],
                               self._vec_eye(wid_s)])

    def gets(self, ym):
        return self._trie[ym]
        #return [wid_b.encode() for wid_b in self._trie[ym]]

    def build_word_tree(self, in_text):
        word_set = [[] for _ in range(len(in_text))]
        for i in range(len(in_text)):
            for prefix in self._trie.prefixes(in_text[i:]):
                word_set[i+len(prefix)-1].append(prefix)
        return word_set


class DicBuilder:
    _dic_csv = [
        "small_lex.csv",
        "core_lex.csv",
        "notcore_lex.csv"
    ]

    def __init__(self, db_path="./anya-dic.db", sudachidic_path="./sudachidic", word_vec_size=8):
        self._db_path = db_path
        self._sudachidic_path = sudachidic_path
        self._word_vec_size = word_vec_size

    def build(self):
        # db initialize
        if os.path.isfile(self._db_path):
            os.remove(self._db_path)

        pos_id_def = os.path.dirname(__file__) + "/data/sudachidic.yml"
        with open(pos_id_def, 'r') as f:
            pos_ids = yaml.safe_load(f)["pos-ids"]

        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()

        cur.execute("CREATE TABLE positions(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING);")
        for pos_name in pos_ids:
            cur.execute('INSERT INTO positions(name) values(?);', (pos_name,))

        cur.execute("""
            CREATE TABLE words(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING, read STRING, pos INTEGER, vec BLOB);
            """)

        for csv_file in self._dic_csv:
            print(" reading... %s" % csv_file)
            with open(self._sudachidic_path + "/" + csv_file) as f:
                reader = csv.reader(f, delimiter=",")
                try:
                    for row in reader:
                        pos_name = row[5] + "." + row[6] + "." + row[7] + "." + row[8]
                        cur.execute("SELECT id FROM positions WHERE name = ?", (pos_name,))
                        pos_id = cur.fetchone()[0]
                        vec_np = np.random.rand(self._word_vec_size)
                        vec = struct.pack('=%df' % vec_np.size, *vec_np)
                        cur.execute("INSERT INTO words(name, read, pos, vec) values(?, ?, ?, ?);",
                                    (row[0], jaconv.kata2hira(row[11]), pos_id, vec))
                except UnicodeDecodeError:
                    pass

        cur.execute('CREATE INDEX words_idx ON words(name, read);')
        cur.execute('commit;')
        cur.close()
        conn.close()

    def insert_vec_from_fasttext(self, fasttext_model="./anya-fasttext.vec"):
        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()
        update_cnt = 0
        csv.field_size_limit(1000000000)

        with open(fasttext_model) as f:
            reader = csv.reader(f, delimiter=" ")
            for i, row in enumerate(reader):
                if i > 0:
                    vec = np.array([float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                                    float(row[5]), float(row[6]), float(row[7]), float(row[8])], dtype=np.float16)
                    vec_pack = struct.pack('=8f', *vec)

                    cur.execute("SELECT id FROM words WHERE name = ?", (row[0],))
                    for data in cur.fetchall():
                        wid = data[0]
                        cur.execute("UPDATE words SET vec = ? WHERE id = ?;", (vec_pack, wid,))
                        update_cnt += 1

        cur.execute('commit;')
        cur.close()
        conn.close()

        print("OK. update_cnt = %d" % update_cnt)

    def insert_bos(self):
        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()

        cur.execute("SELECT id FROM positions WHERE name = ?", ("BOS.*.*.*",))
        pos_id = cur.fetchone()[0]
        vec_np = np.random.rand(self._word_vec_size)
        vec = struct.pack('=%df' % vec_np.size, *vec_np)
        cur.execute("INSERT INTO words(name, read, pos, vec) values(?, ?, ?, ?);", ("_BOS", "_BOS", pos_id, vec))

        cur.execute('commit;')
        cur.close()
        conn.close()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--db_path', help='dictionary DB path', default="./anya-dic.db")
    args = arg_parser.parse_args()
    builder = DicBuilder(args.db_path)
    builder.insert_bos()


if __name__ == "__main__":
    main()

