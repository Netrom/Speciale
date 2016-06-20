__author__ = 'Morten'
import csv

count_b = 0
with open('C:/Users/Morten/Downloads/dsenames2out.csv', 'w', newline='') as out:
    r = csv.reader(open('C:/Users/Morten/Downloads/dsenames.csv'), delimiter=',')
    iter_r = iter(r)
    next(iter_r)
    for i in iter_r:
        count_b += 1
        line = []

        # According to WRDS this is an numeric value. However, sometimes there will be letter.
        #for b in [7, 24, 40, 42]:
        #    if i[b].isalpha():
        #        i[b] = '\\N'

        line.append(count_b)
        for x in i:
            if not x:
                line.append('\\N')
            else:
                line.append(x)

        csv_out = csv.writer(out)
        csv_out.writerow(line)
