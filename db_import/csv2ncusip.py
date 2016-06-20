__author__ = 'Morten'
import csv
r = csv.reader(open('C:/Users/Morten/Downloads/ncusip.csv'))
new_csv = []

iter_r = iter(r)
next(iter_r)


with open('C:/Users/Morten/Downloads/ncusipout.csv', 'w', newline='') as out:
    csv_out = csv.writer(out)
    count = 0
    for line in iter_r:
        count += 1
        if not line[2]:
            # No reason to process data without a PK
            continue
        else:
            # We need the six digits from the cusip and the permco
            new_line = [line[2][:-2].lstrip("0"), line[4], line[1], line[0]]

            # Sometimes ticket will be empty
            if line[3]:
                new_line.append(line[3])
            else:
                new_line.append("\\N")
            csv_out.writerow(new_line)

        # NCUSIP (PK)
        # PERMCO
        # NAME
        # TICKER
        # START_DATE

