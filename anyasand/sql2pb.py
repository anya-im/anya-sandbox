import sqlite3
import struct
import numpy as np
from anyasand.pb.anya_pb2 import Dictionary, Pos, Word


class Sql2Pb:
    def __init__(self, db_path="./anya-dic.db", out_path="./anya-dic.pb", vec_size=8):
        self._db_path = db_path
        self._out_path = out_path
        self._vec_size = vec_size

    def __call__(self):
        dictionary = Dictionary()

        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()

        print("1: Positions")
        cur.execute("SELECT id, name FROM positions;")
        for data in cur.fetchall():
            pos = Pos()
            pos.id = data[0]
            pos.name = data[1]
            dictionary.positions.append(pos)

        print("2: Words")
        cur.execute("SELECT name, read, pos, cost, vec FROM words;")
        for data in cur.fetchall():
            word = Word()
            word.name = data[0]
            word.read = data[1]
            word.pos_id = data[2]
            word.cost = data[3]
            vec_ary = np.array(struct.unpack('=%df' % self._vec_size, data[4]), dtype="float32").tolist()
            for p in vec_ary:
                word.vec.append(p)
            dictionary.words.append(word)

        cur.close()
        conn.close()

        print("3: Serialize")
        with open(self._out_path, "wb") as f:
            try:
                f.write(dictionary.SerializeToString())
            except IOError as e:
                print(e)


def main():
    sql2pb = Sql2Pb()
    sql2pb()


if __name__ == "__main__":
    main()
