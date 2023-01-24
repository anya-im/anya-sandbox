import csv
import os
import sqlite3
import logging
import argparse
import struct
import marisa_trie
import numpy as np
from abc import ABCMeta


class Dictionary(metaclass=ABCMeta):
    def __init__(self, db_path="./anya-dic.db", vec_path=None, vec_size=4, initialize=False):
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


class DicBuilder(Dictionary):
    _dic_csv = [
        "Assert.csv",
        "AuxV.csv",
        "ContentW.csv",
        "Demonstrative.csv",
        "Noun.hukusi.csv",
        "Noun.keishiki.csv",
        "Noun.koyuu.csv",
        "Noun.suusi.csv",
        "Postp.csv",
        "Prefix.csv",
        "Special.csv",
        "Suffix.csv",
        "Rengo.csv"
    ]

    def __init__(self, db_path="./anya-dic.db", vec_path="./anya-fasttext.vec"):
        # db initialize
        if os.path.isfile(db_path):
            os.remove(db_path)

        super().__init__(db_path, vec_path, initialize=True)

    def build(self, in_dic_path, dic_type="juman"):
        if dic_type == "juman":
            # create sql table
            self._cur.execute('CREATE TABLE positions(id INTEGER PRIMARY KEY AUTOINCREMENT, group_id INTEGER, sub_name STRING);')
            self._cur.execute('CREATE TABLE positions_group(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING);')

            # build positions table
            with open(in_dic_path + "/pos-id.def") as f:
                pos_table = {}
                reader = csv.reader(f, delimiter=" ")
                for row in reader:
                    pos = row[0].split(",")
                    if len(pos) >= 2:
                        if not pos[0] in pos_table:
                            pos_table[pos[0]] = []
                        pos_table[pos[0]].append(pos[1])

            for parent, val in pos_table.items():
                for sub in val:
                    self._insert_pid(parent, sub)

            # build words table
            self._cur.execute('CREATE TABLE words(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING, read STRING, pos INTEGER, vec BLOB);')
            for csv_file in self._dic_csv:
                with open(in_dic_path + "/" + csv_file) as f:
                    reader = csv.reader(f, delimiter=",")
                    try:
                        for row in reader:
                            pos_id = self._pid_from_name(row[4], row[5])
                            self._insert_word(row[0], row[9], pos_id)
                            logging.info("word : %s / %s (%d)" % (row[0], row[9], pos_id))
                    except UnicodeDecodeError:
                        pass

            # insert BOS
            self._insert_pid("_BOS", "_BOS")
            pos_id = self._pid_from_name("_BOS", "_BOS")
            self._insert_word("_BOS", "_BOS", pos_id)

            # create index
            self._cur.execute('CREATE INDEX words_idx ON words(name, read);')
            self._is_update = True

        else:
            logging.error("not support dic_type: %s", dic_type)

        self.commit()
        self.close()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--db_path', help='dictionary DB path', default="./anya-dic.db")
    args = arg_parser.parse_args()
    builder = DicBuilder(args.db_path)
    builder.build("/usr/share/mecab/dic/juman/")


if __name__ == "__main__":
    main()

