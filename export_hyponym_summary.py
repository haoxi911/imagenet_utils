import os
import sys
import csv

if __name__ == '__main__':
    if not (len(sys.argv) == 2):
        print("Usage: export_hyponym_summary.py data_folder")

    if len(sys.argv) == 2:
        folders = [name for name in os.listdir(sys.argv[1]) if
                os.path.isdir(os.path.join(sys.argv[1], name) and len(name) > 10 and name[:1] == 'n')]
        for folder in folders:
            wnid = folder[:9]
            name = folder[10:]
            folder = os.path.join(sys.argv[1], folder)
            count = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
            print('%s \t %d \t %s' % (wnid, count, name))
