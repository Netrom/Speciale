__author__ = 'Morten'
import csv

# C:\Users\Morten\Downloads

r = csv.reader(open('C:/Users/Morten/Downloads/ccmxpf_lnkhist.csv'))
new_csv = []

iter_r = iter(r)
next(iter_r)

with open('C:/Users/Morten/Downloads/linkhist2out.csv', 'w', newline='') as out:
    csv_out = csv.writer(out)
    count = 1
    for line in iter_r:
        new_line = []
        if line[7] == "E":
            line[7] = "\\N"
        for i in line:
            if not i:
                i = "\\N"
            new_line.append(i)
        csv_out.writerow(new_line)
