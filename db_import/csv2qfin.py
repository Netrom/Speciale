__author__ = 'Morten'


import csv
import math
r = csv.reader(open('C:/Users/Morten/Downloads/fin1990.csv'))
new_csv = []
q_start = ["not gonna happen", "/01/01", "/04/01", "/07/01", "/10/01"]

iter_r = iter(r)
next(iter_r)

with open('C:/Users/Morten/Downloads/fin1990out.csv', 'w', newline='') as out:
    csv_out = csv.writer(out)
    count = 1
    for line in iter_r:
        count += 1
        new_line = []
        # Make a date variable
        if line[11]:
            year = line[11][0:4]
            q = int(line[11][5:6])
            date = str(year) + q_start[q]
        else:
            # If row 11 is blank it is because the company has changed fiscal year
            # A Q-10 can only have one year but a year can have multiple fiscal years
            # Compustat leaves the line blank (So we know it is a duplicate)
            # Drop it
            continue

        # The SQL Server does not like "" replace it with "\N" (None)
        for i in line:
            if not i:
                i = "\\N"
            new_line.append(i)

        new_line.append(date)
        csv_out.writerow(new_line)


