import csv, re
import os


class CsvMethods:
    def __init__(self, fullname, delimiter):
        filename, extension = os.path.splitext(fullname)
        self.filename = filename
        self.ext = extension
        self.data = csv.reader(open(fullname), delimiter=delimiter)

    def x_row(self, start, rows):
        x = rows
        # Very important that the output is unicode, else MySQL wont work the data from Census "Hagåtña District"
        with open(self.filename + '_' + str(x) + '.csv', 'w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out)
            count = 0
            iter_r = iter(self.data)
            while start > 0:
                next(iter_r)
                start -= 1
            for i in iter_r:
                count += 1
                list_a = []
                count_b = 1
                for a in i:
                    count_b += 1
                    if count_b >= 863:
                        continue
                    if a and a != "N":
                        list_a.append(a)
                    else:
                        list_a.append('\\N')
                # writer.writerow([count] + list_a)
                writer.writerow(list_a)
                if x and count >= x:
                    break

    def schema(self, x):
        data = csv.reader(open(self.filename + '_schema.csv'''))
        count = 0
        list_a = []
        a_quarter = False
        # The schema contains a heading so minus 1..
        for i in data:
            if i[1] == '1':
                list_a.append(count - 1)
            if i[0] == 'DATACQTR':
                a_quarter = count - 1
            count += 1

        iter_r = iter(self.data)
        next(iter_r)

        d_quar = ['0101', '0301', '0601', '0901']
        with open(self.filename + '_' + str(x) + '_schema.csv', 'w', newline='') as out:
            csv_out = csv.writer(out)
            count = 0
            for i in iter_r:
                value = i[a_quarter]
                if not value:
                    # If row actual date is blank it is because the company has changed fiscal year
                    # A Q-10 can only have one year but a year can have multiple fiscal years
                    # Compustat leaves it blank (So we know it is a duplicate) -> Drop it
                    continue

                year = value[0:4]
                q = int(value[5:6]) - 1
                q_start = year + d_quar[q]
                list_b = [count, q_start]

                for b in list_a:
                    value = i[b]
                    if value:
                        list_b.append(value)
                    else:
                        list_b.append('\\N')

                csv_out.writerow(list_b)
                count += 1

                if x and count >= x:
                    break

    def table_creator(self, name):
        schema = csv.reader(open(self.filename + '_schema.csv'''))
        # Read the first row
        sql0 = 'CREATE TABLE ' + name + ' ('

        for i in self.data:
            row = i
            break

        key_lookup = {}
        for i in schema:
            key_lookup[i[0]] = i[2]

        for i in row:
            type = self.mysql_translater(key_lookup[i])
            variable = i + ' ' + type + ', '
            sql0 += variable

        print(sql0)

    @staticmethod
    def mysql_translater(key):
        if key == 'date':
            m_key = 'date'
        elif key == 'integer':
            m_key = 'int(10)'
        else:
            re01 = '([a-z]+)'
            re02 = '(\\(.*?\\))'
            rg = re.compile(re01+re02)
            data = rg.search(key)
            name = data.group(1)
            number = data.group(2)

            if name == 'char':
                m_key = 'varchar' + number
            elif name == 'numeric':
                m_key = 'decimal' + number
            else:
                raise ValueError('Unable to determinate key')
        return m_key


def funda():
    file = 'C:\\Users\\Morten\Downloads\\fund\\funda'
    csv_file = CsvMethods(file, ',')
    csv_file.x_row(1, False)
    # csv_file.schema(False)
    # csv_file.table_creator('funda_test')


def census():
    file = 'C:/Users/Morten/Downloads/CB0700CZ2/CB0700CZ2.dat'
    csv_file = CsvMethods(file, ',')
    csv_file.x_row(0, 500)


def onetofive(filename):
    data = csv.reader(open(filename))
    with open(filename + '_five.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        for i in data:
            base = [i[0], '\\N', '\\N']
            csv_out.writerow(i[:3] + ['100', '\\N'])
            for x in i[3:6]:
                csv_out.writerow(base+[x, '\\N'])
            csv_out.writerow(base + [i[6], i[7]])

if __name__ == "__main__":
    #funda()
    #census()
    #onetofive('C:/Users/Morten/Downloads/2007/1997.31.csv')
    c = CsvMethods('C:/Users/Morten/Downloads/share_out.csv', ',')
    c.x_row(0, False)






