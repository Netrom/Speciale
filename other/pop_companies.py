from database import dbconn
import csv
__author__ = 'Morten'


def table_indu_stat(conn):
    sql_query0=\
        "SELECT DISTINCT b.siccd, y.num "\
        "FROM dsenames as b "\
        "LEFT JOIN "\
        "	(SELECT d.siccd, count(d.prim) as num "\
        "	FROM "\
        "		(SELECT prim, permco, siccd, MAX(namedt) as m_namedt "\
        "		FROM dsenames "\
        "		WHERE namendt >= %s and %s >= namedt "\
        "		GROUP BY permco, siccd) as d "\
        "	GROUP BY d.siccd) AS y "\
        "ON b.siccd = y.siccd "\
        "ORDER BY siccd ASC"

    writer = csv.writer(open('C:/Users/Morten/Downloads/test_table.csv', 'w', newline=''))

    list_a = []
    list_y = []
    cur0 = conn.cur_buff()
    year = 1998
    while year <= 2014:
        list_y.append(year)
        y_date = str(year) + '-12-31'
        cur0.execute(sql_query0, (y_date, y_date))

        list_b = []
        if year == 1998:
            list_c = []
            for i in cur0:
                list_c.append(i[0])
                list_b.append(i[1])
            list_a.append(list_c)
            list_a.append(list_b)
        else:
            for i in cur0:
                list_b.append(i[1])
            list_a.append(list_b)

        year += 1

    writer.writerow(['sic'] + list_y)
    for b in zip(*list_a):
        writer.writerow(b)


def main():
    conn = dbconn()
    table_indu_stat(conn)


if __name__ == "__main__":
    main()
