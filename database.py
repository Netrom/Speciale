__author__ = 'Morten'
import mysql.connector
from mysql.connector import errorcode

def configs(db):
    config = dict(
        speciale=dict(
            host='localhost',
            database='speciale',
            user='speciale',
            password='python',
            raise_on_warnings=True,)
        )
    try:
        cfg = config[db]
    except KeyError:
        print('The database {0} is not yest configured'.format(db))
    return cfg


class SqlDB:
    def __init__(self, config):
        config = configs(config)
        try:
            self.cnx = mysql.connector.connect(**config)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("DB: Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("DB: Database does not exist")
            else:
                print(err)
            raise

    def __exit__(self):
        if self.cnx:
            self.cnx.close()

    def cursor(self):
        return self.cnx.cursor()

    def insert(self, sql, args):
        cur0 = self.cnx.cursor()
        cur0.execute(sql, args)
        self.cnx.commit()
        cur0.close()

    def cur_buff(self):
        return self.cnx.cursor(buffered=True)

    def truncate(self, table):
        sql0 = "TRUNCATE {0}".format(table)
        cur0 = self.cnx.cursor()
        cur0.execute(sql0)
        cur0.close()

    def fetch_dict(self, sql, args=None):
        result = []
        cur = self.cur_buff()
        if args:
            try:
                cur.execute(sql, args)
            except:
                print(cur.statement)
                raise
        else:
            cur.execute(sql)
        columns = tuple([d[0] for d in cur.description])
        for row in cur:
            result.append(dict(zip(columns, row)))
        cur.close()
        return result

    def fetch_one(self, sql, args=None):
        result = []
        cur = self.cur_buff()
        if args:
            try:
                cur.execute(sql, args)
            except:
                print(cur.statement)
                raise
        else:
            cur.execute(sql)
        columns = tuple([d[0] for d in cur.description])
        for row in cur:
            result.append(dict(zip(columns, row)))
        if len(result) == 1:
            result = result[0]
        elif len(result) > 1:
            raise IndexError('More than one result')
        else:
            cur.close()
            return None
        cur.close()
        return result

    #table_indu_stat(conn)
    #loop(conn)
