from __future__ import print_function
from datetime import timedelta
import operator

import mysql.connector
from mysql.connector import errorcode
import statsmodels.api as sm
import numpy as np

config = dict(host='localhost',
              database='speciale',
              user='speciale',
              password='python',
              raise_on_warnings=True,)


class SqlDB:
    def __init__(self):
        try:
            self.cnx = mysql.connector.connect(**config)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)
            self.cnx = None

    def close(self):
        if self.cnx:
            self.cnx.close()

    def cur_buff(self):
        return self.cnx.cursor(buffered=True)


def create_ipgv(c):
    n = c.cnx.cursor()
    n.execute("INSERT INTO ipgv "
              "SELECT DISTINCT gvkey, cusip "
              "FROM pstock")
    c.cnx.commit()


# Match gvkeys
def first_round(c, sql_query0, sql_query1):
    cursor = c.cnx.cursor(buffered=True)
    cursor.execute(sql_query0)
    for i in cursor:
        cursor2 = c.cnx.cursor()
        cursor2.execute(sql_query1, (i[0], i[1]))
        c.cnx.commit()


def lookup_tar(c):
    sql_query0 = \
        "SELECT i.gvkey, d.deal_no "\
        "FROM deals as d "\
        "INNER JOIN ipgv as i "\
        "ON d.tar_cusip = i.cusip"

    sql_query1 =\
        "UPDATE deals " \
        "SET tar_gvkey=%s " \
        "WHERE deal_no=%s"

    # First round
    first_round(c, sql_query0, sql_query1)

    sql_query0 =\
        "SELECT deal_no, tar_name, tar_cusip, date_announced "\
        "FROM deals "\
        "WHERE tar_gvkey is null and date_announced > '1990-01-01' and acq_type = 'CORPORATE'"

    sql_query2 =\
        "SELECT ipgv.gvkey, ncusip.name "\
        "FROM ipgv RIGHT JOIN ncusip ON ncusip.ncusip = ipgv.cusip "\
        "WHERE cusip IN ("\
        "SELECT ncusip FROM ncusip WHERE permco IN ("\
        "SELECT permco FROM ncusip WHERE ncusip=%s))"

    # Second round
    helper(c, sql_query0, sql_query1, sql_query2)

    sql_query2 =\
        "SELECT l.gvkey, n.name " \
        "FROM ncusip as n  " \
        "INNER JOIN linktbl as l on n.permco = l.permco " \
        "WHERE n.ncusip=%s"

    # Third round
    helper(c, sql_query0, sql_query1, sql_query2)


def lookup_acq(c):
    sql_query0 = \
        "SELECT i.gvkey, d.deal_no "\
        "FROM deals as d "\
        "INNER JOIN ipgv as i "\
        "ON d.acq_cusip = i.cusip"

    sql_query1 =\
        "UPDATE deals " \
        "SET acq_gvkey=%s " \
        "WHERE deal_no=%s"

    # First round
    first_round(c, sql_query0, sql_query1)

    sql_query0 =\
        "SELECT deal_no, acq_name, acq_cusip, date_announced "\
        "FROM deals "\
        "WHERE acq_gvkey is null and date_announced > '1990-01-01' and acq_type = 'CORPORATE'"

    sql_query2 =\
        "SELECT ipgv.gvkey, ncusip.name "\
        "FROM ipgv RIGHT JOIN ncusip ON ncusip.ncusip = ipgv.cusip "\
        "WHERE cusip IN ("\
        "SELECT ncusip FROM ncusip WHERE permco IN ("\
        "SELECT permco FROM ncusip WHERE ncusip=%s))"

    # Second round
    helper(c, sql_query0, sql_query1, sql_query2)

    sql_query2 =\
        "SELECT l.gvkey, n.name " \
        "FROM ncusip as n  " \
        "INNER JOIN linktbl as l on n.permco = l.permco " \
        "WHERE n.ncusip=%s"

    # Third round
    helper(c, sql_query0, sql_query1, sql_query2)


def helper(c, sql_query0, sql_query1, sql_query2):
    cur0 = c.cnx.cursor(buffered=True)
    cur0.execute(sql_query0)

    for i in cur0:
        deal_no = i[0]
        deal_name = i[1]
        cusip = i[2]
        date_announced = i[3]

        cur1 = c.cnx.cursor(buffered=True)
        cur1.execute(sql_query2, (cusip,))
        gvkey_count = cur1.rowcount
        if gvkey_count >= 1:
            cur2 = c.cnx.cursor(buffered=True)
            for b in cur1:
                gvkey = b[0]
                ncusip_name = b[1]
                # Check gvkey and announcement date against computerstat, solves two problems
                # 1: One permco can have multiple gvkeys,
                # but for a particular stock date one permco can only have one gvkey
                # 2: Wrong cusip in the dataset.
                sql_query3 =\
                    "SELECT gvkey " \
                    "FROM pstock " \
                    "WHERE datadate BETWEEN %s and %s and gvkey=%s"
                start_time = date_announced - timedelta(days=10)
                cur2.execute(sql_query3, (start_time, date_announced, gvkey))

                # Obviously there can exist multiple gvkeys in the 10 days
                # This however is not a problem, as we will catch em later on
                # When the 250 trading days filter is applied

                if cur2.rowcount > 1:
                    cur2 = c.cnx.cursor()
                    cur2.execute(sql_query1, (gvkey, deal_no))
                    c.cnx.commit()
                    print(str(deal_no) + ": " + deal_name + "==" + ncusip_name)
                    break


# Get estimated returns
class AbnormalReturnData:
    def __init__(self, conn, date):
        t0 = -250
        t1 = -51
        t2 = -5
        t3 = 5
        t_array = [t0, t0-1, t1, t2, t2-1, t3]

        # Day zero is the announcement date = (c1-c0)/c0-1
        # If the announcement date is not a trading day, it is the first day after announcement
        sql_query0=\
            "SELECT ff.date " \
            "FROM ff " \
            "WHERE ff.date>=%s "\
            "ORDER by ff.date ASC " \
            "LIMIT 1"
        cur2 = conn.cur_buff()
        cur2.execute(sql_query0, (date,))

        dates = {}
        for i in cur2:
            date_0 = i[0]
        # Divides the t_array into positive and negative
        neg = []
        pos = []
        for i in t_array:
            if i > 0:
                pos.append(i)
            elif i < 0:
                neg.append(i)

        # If the postive list has values, start looking those values up
        if len(pos) > 0:
            sql_query1 = \
                "SELECT ff.date "\
                "FROM ff " \
                "WHERE ff.date>%s " \
                "ORDER BY ff.date ASC " \
                "LIMIT %s"
            cur2.execute(sql_query1, (date_0, max(pos)))
            count = 0
            for i in cur2:
                count += 1
                if count in pos:
                    dates[count] = i[0]

        # If the negative list has values, start looking those values up
        if len(neg) > 0:
            sql_query1 = \
                "SELECT ff.date "\
                "FROM ff " \
                "WHERE ff.date<%s " \
                "ORDER BY ff.date DESC " \
                "LIMIT %s"
            cur2.execute(sql_query1, (date_0, -min(neg)))
            count = 0
            for i in cur2:
                count -= 1
                if count in neg:
                    dates[count] = i[0]

            self.conn = conn
            self.date = date
            self.t0 = dates[t0]
            self.t0m1 = dates[t0-1]
            self.t1 = dates[t1]
            self.t2 = dates[t2]
            self.t2m1 = dates[t2-1]
            self.t3 = dates[t3]
            # Default is the model window!
            # It can be changed using the estimation_window fucntion

    def other_bids(self, gvkey):
        # Test acquirers that have made bids during the stimation period
        sql_query1=\
            "SELECT date_announced, date_effective " \
            "FROM deals " \
            "WHERE acq_gvkey=%s AND date_announced BETWEEN %s AND %s"
        cur0 = self.conn.cur_buff()
        cur0.execute(sql_query1, (gvkey, self.t0, self.t3))

        if cur0.rowcount > 1:
            return True
        return False

    def primary_stock(self, gvkey):
        # This function returns the primary stock iid for the period
        sql_query0 = \
            "SELECT iid " \
            "FROM pstock " \
            "WHERE gvkey=%s AND datadate=%s"
        cur0 = self.conn.cur_buff()
        cur0.execute(sql_query0, (gvkey, self.t0m1))

        sql_query1 = \
            "SELECT volume, pclose " \
            "FROM pstock " \
            "WHERE gvkey=%s AND iid=%s AND datadate BETWEEN %s AND %s" \

        iid = False
        obs_high = 0
        median_high = 0
        avg_high = 0
        print(self.t0m1)
        print(self.t1)
        for i in cur0:
            cur0.execute(sql_query1, (gvkey, i[0], self.t0m1, self.t3))
            obs = 0
            trade = []
            for x in cur0:
                obs += 1
                trade.append(float(x[0])*float(x[1]))
            median = np.median(trade)
            avg = np.average(trade)
            if obs > obs_high:
                iid = i[0]
            elif obs == obs_high:
                if median > median_high:
                    median_high = median
                    iid = i[0]
                elif median_high == 0 and avg > avg_high:
                    avg_high = avg
                    iid = i[0]
        print(iid)
        return iid

        # We only have common stocks in the DB, so no filter is neceassery
        # If multiple cs exist, we should just use the one with the highest trading?

    def exogenous_estimation(self):
        return self.exogenous(self.t0, self.t1)

    def exogenous_model(self):
        return self.exogenous(self.t2, self.t3)

    def exogenous(self, begin, end):
            # Lookup some keys
            sql_query0 =\
                "SELECT ff.date, ff.rf, ff.mprem, ff.smb, ff.hml, ff.rmw, ff.cma "\
                "FROM ff "\
                "WHERE ff.date BETWEEN %s and %s "\
                "ORDER by ff.date ASC "\

            # We need 251 as we only have prices, and not % change, so we need the primo value
            cur0 = self.conn.cur_buff()
            cur0.execute(sql_query0, (begin, end))
            capm = []
            ff3 = []
            ff5 = []
            mm = []
            rf = []
            for i in cur0:
                rft = float(i[1])
                premium = float(i[2])
                smb = float(i[3])
                hml = float(i[4])
                rmw = float(i[5])
                cma = float(i[6])
                # Exogenous
                mm.append([1, rft+premium])
                capm.append([premium])
                ff3.append([premium, smb, hml])
                ff5.append([premium, smb, hml, rmw, cma])
                rf.append(rft)

            return dict(mm=mm, capm=capm, ff3=ff3, ff5=ff5, rf=rf)

    def endogenous_estimation(self, gvkey, iid):
        return self.endogenous(gvkey, iid, self.t0m1, self.t1)

    def endogenous_model(self, gvkey, iid):
        return self.endogenous(gvkey, iid, self.t2m1, self.t3)

    def endogenous(self, gvkey, iid, begin, end):
        cur0 = self.conn.cur_buff()
        sql_query0 = \
            "SELECT datadate, pclose, trfd " \
            "FROM pstock " \
            "WHERE gvkey=%s AND iid=%s AND datadate BETWEEN %s AND %s " \
            "ORDER BY datadate ASC"
        cur0.execute(sql_query0, (gvkey, iid, begin, end))

        s_returns = []
        count = 0
        for i in cur0:
            now = (float(i[1]))*float(i[2])
            if 0 < count:
                s_return = (now/prev-1)*100
                s_returns.append(s_return)
            else:
                count += 1
            prev = now
        return s_returns


class EventData:
    def __init__(self, conn, date):
        t = [-250, -51, -1, 1]
        dates = self.dates(conn, date, t)
        self.t0 = dates[t[0]]
        self.t0m1 = dates[(t[0]-1)]
        self.t1 = dates[t[1]]
        self.t2 = dates[t[2]]
        self.t2m1 = dates[(t[2]-1)]
        self.t3 = dates[t[3]]
        self.event = dates[0]
        self.conn = conn
        self.t = t

    def other_acq_bids(self, gvkey):
        # Test acquirers that have made bids during the stimation period
        sql_query1=\
            "SELECT date_announced, date_effective " \
            "FROM deals " \
            "WHERE acq_gvkey=%s AND date_announced BETWEEN %s AND %s"
        cur0 = self.conn.cur_buff()
        cur0.execute(sql_query1, (gvkey, self.t0m1, self.t3))

        if cur0.rowcount > 1:
            return True
        return False

    def primary_stock(self, gvkey):
        # This function returns the primary stock iid for the period
        sql_query0 = \
            "SELECT iid " \
            "FROM pstock " \
            "WHERE gvkey=%s AND datadate=%s"
        cur0 = self.conn.cur_buff()
        cur0.execute(sql_query0, (gvkey, self.event))

        sql_query1 = \
            "SELECT volume, pclose " \
            "FROM pstock " \
            "WHERE gvkey=%s AND iid=%s AND datadate BETWEEN %s AND %s" \

        median_high = 0
        avg_high = 0

        cur1 = self.conn.cur_buff()
        # Loop through all the iid
        for i in cur0:
            cur1.execute(sql_query1, (gvkey, i[0], self.t0m1, self.event))
            obs = 0
            trade = []
            for x in cur0:
                obs += 1
                trade.append(float(x[0])*float(x[1]))
            median = np.median(trade)
            avg = np.average(trade)
            freedoms = 2
            # Remember that 0 is also an obs, so we do > and not >=
            if obs > (-self.t0m1+self.t3-freedoms):
                if median > median_high:
                    median_high = median
                    iid = i[0]
                elif median_high == 0 and avg > avg_high:
                    avg_high = avg
                    iid = i[0]

        return iid

        # We only have common stocks in the DB, so no filter is neceassery
        # If multiple cs exist, we should just use the one with the highest trading?

    def model_data(self, gvkey, iid):
        model_data = self.maynes_approach(self.conn, gvkey, iid, self.t0m1, self.t1, self.t[0])
        model_data = homogen(model_data)
        params = sm.OLS(model_data['endogenous'], model_data['exogenous']).fit().params
        event_data = self.maynes_approach(self.conn, gvkey, iid, self.t2m1, self.t3, self.t[2])
        return dict(params=params, event_data=event_data)

    @staticmethod
    def dates(conn, date, t_array):
        # Day zero is the announcement date = (c1-c0)/c0-1
        # If the announcement date is not a trading day, it is the first day after announcement
        sql_query0=\
            "SELECT ff.date " \
            "FROM ff " \
            "WHERE ff.date>=%s "\
            "ORDER by ff.date ASC " \
            "LIMIT 1"
        cur2 = conn.cur_buff()
        cur2.execute(sql_query0, (date,))

        dates = {}
        for i in cur2:
            date_0 = i[0]
        # Divides the t_array into positive and negative
        neg = []
        pos = []
        for i in t_array:
            if i > 0:
                pos.append(i)
            elif i < 0:
                neg.append(i)

        # If the postive list has values, start looking those values up
        if len(pos) > 0:
            sql_query1 = \
                "SELECT ff.date "\
                "FROM ff " \
                "WHERE ff.date>%s " \
                "ORDER BY ff.date ASC " \
                "LIMIT %s"
            cur2.execute(sql_query1, (date_0, max(pos)))
            count = 0
            for i in cur2:
                count += 1
                dates[count] = i[0]

        # If the negative list has values, start looking those values up
        if len(neg) > 0:
            sql_query1 = \
                "SELECT ff.date "\
                "FROM ff " \
                "WHERE ff.date<%s " \
                "ORDER BY ff.date DESC " \
                "LIMIT %s"
            cur2.execute(sql_query1, (date_0, -min(neg)))
            count = 0
            for i in cur2:
                count -= 1
                dates[count] = i[0]

        return dates

            # Default is the model window!
            # It can be changed using the estimation_window fucntion

    @staticmethod
    def maynes_approach(conn, gvkey, iid, begin, end, t_start):
        # Gvkey, iid, begin, end
        cur0 = conn.cur_buff()
        # There can be missing stock data in Compustat.
        # Left joining the FF dataset solves this.
        sql_query0 = \
            "SELECT ff.date, ff.mprem, ff.rf, s.pclose, s.pstatus, s.trfd "\
            "FROM ff LEFT JOIN (" \
                "SELECT pclose, pstatus, datadate, trfd " \
                "FROM speciale.pstock " \
                "WHERE gvkey = 15759 and iid='01') as s " \
            "ON ff.date = s.datadate "\
            "WHERE ff.date BETWEEN '1991-03-05' and '1991-12-17' "\
            "ORDER BY ff.date ASC"
        #cur0.execute(sql_query0)
        sql_query0 = \
            "SELECT ff.date, ff.mprem, ff.rf, s.pclose, s.pstatus, s.trfd "\
            "FROM ff LEFT JOIN (" \
                "SELECT pclose, pstatus, datadate, trfd " \
                "FROM speciale.pstock " \
                "WHERE gvkey=%s and iid=%s) as s " \
            "ON ff.date = s.datadate "\
            "WHERE ff.date BETWEEN %s and %s "\
            "ORDER BY ff.date ASC"
        cur0.execute(sql_query0, (gvkey, iid, begin, end))

        prev_count = None
        prev_close = None
        rm_cache = []
        endogenous = []
        exogenous = []
        t = []
        #
        count = 0
        for i in cur0:
            count += 1
            status = i[4]
            market = float(i[1])+float(i[2])
            # If not status 3 it means that no trade took place that day
            if status == 3:
                adj_close = float(i[3])*float(i[5])
                # A primo is needed before calculations can start
                if prev_count:
                    # The ln of (primo/ultimo)
                    rt = np.log(adj_close/prev_close)*100

                    # Remember that the FF set is in *100 (pct)
                    rmt = sum(rm_cache)+np.log1p(market/100)*100

                    # Add to the lists
                    n = count - prev_count
                    tt = t_start + count - 1
                    t.append(tt)
                    endogenous.append(rt)
                    exogenous.append([n, rmt])

                prev_close = adj_close
                prev_count = count
                rm_cache = []
            elif prev_count:
                rm_cache.append(np.log1p(market/100)*100)

        return dict(t=t, endogenous=endogenous, exogenous=exogenous)


class MergerData(EventData):
    def __init__(self, conn, date, acq_gvkey, tar_gvkey):
        EventData.__init__(self, conn, date)

        acq_iid = self.primary_stock(acq_gvkey)
        if not acq_iid:
            raise ValueError('Unable to determine acq iid')

        tar_iid = self.primary_stock(tar_gvkey)
        if not tar_iid:
            raise ValueError('Unable to determine tar iid')

        if self.other_acq_bids:
            raise ValueError('Acquirer has other bids')

        self.acq_gvkey = acq_gvkey
        self.tar_gvkey = tar_gvkey
        self.acq_iid = acq_iid
        self.tar_iid = tar_iid

    def acq_calc(self):
        acq_data = self.model_data(self.acq_gvkey, self.acq_iid)
        return EventCalc(acq_data)

    def tar_calc(self):
        tar_data = self.model_data(self.tar_gvkey, self.tar_iid)
        return EventCalc(tar_data)


class EventCalc:
    # Takes model_data from the EventData
    def __init__(self, model_data):
        self.event_data = model_data['event_data']
        self.params = model_data['params']

    def abnormal_returns(self):
        return self.abnormal(self.params, self.event_data)

    def abnormal_maynes(self):
        event_data = homogen(self.event_data)
        return self.abnormal(self.params, event_data)

    @staticmethod
    def abnormal(params, event_data):
        abnormals = []
        for end, exog in zip(event_data['endogenous'], event_data['exogenous']):
            abnormal = exog-np.inner(np.array(end), np.array(params))
            abnormals.append(abnormal)
        return dict(t=event_data['t'], returns=abnormals)


def homogen(data):
    no_prev = 0
    endogenous = []
    exogenous = []
    for end, exog in zip(data['endogenous'], data['exogenous']):
        no = exog[0]
        n = no-no_prev
        f = 1/np.sqrt(n)
        exogenous.append(f*np.array(exogenous))
        endogenous.append(f*end)
    return dict(endogenous=endogenous, exogenous=exogenous)


class AbnormalReturn:
    def __init__(self, c, date):
        self.d = AbnormalReturnData(c, date)
        # First we need the data for estimating the params
        self.e_exog = self.d.exogenous_estimation()
        # The data for the estimation period
        self.m_exog = self.d.exogenous_model()

    def primary_stock(self, gvkey):
        return self.d.primary_stock(gvkey)

    def abnormal_return(self, gvkey, iid):
        actual = self.actual_return(gvkey, iid)
        models = self.model_return(gvkey, iid)
        returns = {}
        for model in models:
            m_returns = []
            for a, m in zip(actual, models[model]):
                m_returns.append(a-m)
            returns[model] = m_returns
        return returns

    def actual_return(self, gvkey, iid):
        return self.d.endogenous_model(gvkey, iid)

    def model_return(self, gvkey, iid):
        m_params = self.models_estimator(gvkey, iid)
        e_returns = {}
        for model in m_params:
            params = m_params[model]
            returns = []
            # MM is the only model that is not a economic model with risk-free rate.
            # Easiest solution is just to calculate that one
            if model == 'mm':
                for i in self.m_exog[model]:
                    # i[0] is just the intercept (it is one)
                    returns.append(params[0]+params[1]*i[1])
            else:
                for i, x in zip(self.m_exog[model], self.m_exog['rf']):
                    total = x
                    for b, q in zip(params, i):
                        total += b*q
                    returns.append(total)
            e_returns[model] = returns
        return e_returns
        #print(acq_othbids(conn, '2109995020'))
        #print(acq_othbids(conn, '2031349020'))

    def models_estimator(self, gvkey, iid):
        # _estimation is used for estimating the model, _model is used for forecasting "part"
        s_returns = self.d.endogenous_estimation(gvkey, iid)
        s_premiums = list(map(operator.sub, s_returns, self.e_exog['rf']))
        print(len(s_returns))
        print(len(self.e_exog['mm']))
        mm = sm.OLS(s_returns, self.e_exog['mm']).fit().params
        capm = sm.OLS(s_premiums, self.e_exog['capm']).fit().params
        ff3 = sm.OLS(s_premiums, self.e_exog['ff3']).fit().params
        ff5 = sm.OLS(s_premiums, self.e_exog['ff5']).fit().params
        return dict(mm=mm, capm=capm, ff3=ff3, ff5=ff5)


def loop():
    conn = SqlDB()
    sql_query0 = \
        "SELECT deal_no, date_announced, acq_gvkey, tar_gvkey " \
        "FROM deals " \
        "WHERE acq_nat='US' and tar_nat='US' AND deals.status='C' " \
        "ORDER BY date_announced ASC"

    cur0 = conn.cur_buff()
    cur0.execute(sql_query0)
    for i in cur0:
        print(i[0])
        print(i[1])
        da = i[1]
        acq_gvkey = i[2]
        tar_gvkey = i[3]
        ar = AbnormalReturn(conn, da)
        # Acquire
        acq_iid = ar.primary_stock(acq_gvkey)
        tar_iid = ar.primary_stock(tar_gvkey)
        if not acq_iid or not tar_iid:
            # Insufficient data
            continue

        ar.model_return(acq_gvkey, acq_iid)
        print(ar.model_return(tar_gvkey, tar_iid))

    conn.close()

def test2():
    conn = SqlDB()
    d = AbnormalReturnData(conn, '1992-03-02')
    data=d.maynes_approach()
    import csv
    with open('C:/Users/Morten/Downloads/stockcusout.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        for i in zip(data['no'], data['mm_inter'], data['rt_list'], data['rmt_list']):
            csv_out.writerow(i)


def test():
    conn = SqlDB()
    ar = AbnormalReturn(conn, '1992-03-02')
    gvkey = 15759
    #gvkey = 5250
    iid = ar.primary_stock(gvkey)
    out = ar.abnormal_return(gvkey, iid)
    print(out['ff3'])


if __name__ == "__main__":
    test2()
    #test()
    #lookup_acq(conn)
    #lookup_tar(conn)
    #beta(conn, '2109995020')
    #models(conn)
    #print(trading_date(conn, '2015/06/04', 6))



