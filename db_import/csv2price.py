__author__ = 'Morten'


import csv
r = csv.reader(open('C:/Users/Morten/Downloads/stock1983.csv'))
new_csv = []

iter_r = iter(r)
next(iter_r)

with open('C:/Users/Morten/Downloads/stock1983out.csv', 'w', newline='') as out:
    csv_out = csv.writer(out)
    count = 0
    for line in iter_r:
        #count += 1
        #if count > 6000:
        #   break

        cusip = line[4][0:6]
        line[4] = cusip.lstrip("0")

        new_line = []
        for i in line:
            # The SQL Server does not like "" replace it with "\N" (None)
            if not i:
                i = "\\N"
            new_line.append(i)
        csv_out.writerow(new_line)
