import os
import sys
import csv

if __name__ == '__main__':
    if not (len(sys.argv) == 2):
        print("Usage: export_hyponym_summary.py data_folder")

    if len(sys.argv) == 2:
        data = []
        for root, dirs, files in os.walk(sys.argv[1]):
            if root != sys.argv[1]:
                path = os.path.basename(os.path.normpath(root))
                if path[:1] == "n" and len(path) > 10:
                    data.append((path[:9], len(files), path[10:]))
        with open('summary.csv', 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for wnid, count, name in data:
                writer.writerow([wnid, count, name])
