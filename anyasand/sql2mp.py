import sqlite3
import struct
import msgpack
import numpy as np
from tqdm import tqdm


class Sql2Mp:
    def __init__(self, db_path="./anya-dic.db", out_path="./anya-dic.msg", vec_size=8):
        self._db_path = db_path
        self._out_path = out_path
        self._vec_size = vec_size

    def __call__(self):
        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()

        print("1: Positions")
        cur.execute("SELECT name FROM positions;")
        pos = []
        for data in cur.fetchall():
            pos.append(data[0])

        print("2: Words")
        cur.execute("SELECT name, read, pos, cost, vec FROM words;")
        words = []
        results = cur.fetchall()
        for data in tqdm(results):
            words.append({
                "name": data[0],
                "read": data[1],
                "pos_id": data[2],
                "cost": data[3],
                "vec": np.array(struct.unpack('=%df' % self._vec_size, data[4]), dtype="float16").tolist()
            })

        cur.close()
        conn.close()

        print("3: Serialize")
        with open(self._out_path, "wb") as f:
            f.write(msgpack.packb(pos))
            f.write(msgpack.packb(words))


def main():
    sql2pb = Sql2Mp()
    sql2pb()


if __name__ == "__main__":
    main()
