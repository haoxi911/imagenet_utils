import os
import sys
import csv

if __name__ == '__main__':
    if not (len(sys.argv) == 2):
        print("Usage: export_hyponym_summary.py data_folder")

    if len(sys.argv) == 2:
        for root, dirs, files in os.walk(sys.argv[1]):
            if root[:1] == 'n':
                wnid = root[:9]
                name = root[10:]
                print('%s \t %d \t %s' % (wnid, len(files), name))

