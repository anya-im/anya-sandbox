import csv
import sqlite3
import struct
import numpy as np
from tqdm import tqdm


class Sql2Csv:
    def __init__(self, db_path="./anya-dic.db", out_pos_path="./anya-dic-pos.tsv", out_path="./anya-dic.tsv", vec_size=8):
        self._db_path = db_path
        self._out_pos_path = out_pos_path
        self._out_path = out_path
        self._vec_size = vec_size

    def __call__(self):
        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()

        print("1: Positions")
        with open(self._out_pos_path, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            cur.execute("SELECT name FROM positions;")
            for data in cur.fetchall():
                writer.writerow([data[0]])

        print("2: Words")
        with open(self._out_path, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            cur.execute("SELECT name, read, pos, cost, vec FROM words;")
            results = cur.fetchall()
            for data in tqdm(results):
                vec = np.array(struct.unpack('=%df' % self._vec_size, data[4]), dtype="float16").tolist()
                writer.writerow([data[0], data[1], data[2], data[3],
                                 vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7]])

        cur.close()
        conn.close()


def main():
    sql2pb = Sql2Csv()
    sql2pb()


if __name__ == "__main__":
    main()
