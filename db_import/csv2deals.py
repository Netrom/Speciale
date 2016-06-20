__author__ = 'Morten'
import csv
import os


def date_rewrite(date):
    date = date[6:10] + "/" + date[0:2] + "/" + date[3:5]
    return date


def convert_csv():
    with open('C:/Users/Morten/Downloads/new/out5.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        for root, dirs, files in os.walk('C:/Users/Morten/Downloads/new/csv3'):
            for file in files:
                if file.endswith(".csv"):
                    print(file)
                    r = csv.reader(open(os.path.join(root, file)))
                    iter_r = iter(r)
                    next(iter_r)
                    next(iter_r)
                    next(iter_r)
                    for line in iter_r:
                        csv_out.writerow(convert_line(line))


def convert_line(line):
    if line[32] != '-':
        line[32] = 1
    # Check if naic and sic codes are an integer
    # A few companies seems to have invented their own NAICS/SIC code
    # These are not governed by the standardization and are considered to be unreliable
    # Hence, it is not possible to find industry factor for these companies
    # For some reason I included a fifth sic (column 24), just run the check and drop the column in the DB later
    for i in [1, 2, 31, 34]:
        if line[i] != '-':
            line[i] = date_rewrite(line[i])

    for i in [16, 17, 22, 23, 24]:
        try:
            int(line[i])
        except:
            line[i] = "\\N"

    # Create a bit(1) instead of varchar(x) in DB
    for i in [27, 29]:
        if line[i] == "Public":
            # Any value here will result in 1, even 0...
            # However mysql will cast of a warning, no worries, I checked that the data loaded perfectly
            # before adding this step to the code
            line[i] = "1"
        else:
            # Zero to the mysql is empty
            line[i] = "0"

    new_line = []
    # Dash means null in Thomson...
    for i in line:
        if i == "-":
            i = "\\N"
        new_line.append(i)
    return new_line


def deflat():
    with open('C:/Users/Morten/Downloads/new/out4.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        r = csv.reader(open('C:/Users/Morten/Downloads/new/out.csv'))
        for l in r:
            for t in l[13].splitlines():
                csv_out.writerow([l[0], t])


if __name__ == "__main__":
    convert_csv()
    #deflat()
