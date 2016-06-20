__author__ = 'Morten'
import csv
r = csv.reader(open('C:/Users/Morten/Downloads/linktbl.csv'))
new_csv = []

iter_r = iter(r)
next(iter_r)

with open('C:/Users/Morten/Downloads/linktblout.csv', 'w', newline='') as out:
    csv_out = csv.writer(out)
    count = 1
    for line in iter_r:
        new_line = [line[0], line[3], line[6]]

        csv_out.writerow(new_line)

