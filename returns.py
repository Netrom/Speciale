import statsmodels.api as sm
import scipy.stats as st
import numpy as np
import csv
from database import SqlDB
from itertools import chain
import re
from decimal import *
import collections

__author__ = 'Morten'


class EventData:
    def __init__(self, conn, da, t):
        # Retrieves the trading date of time 0
        self.event = self.trade_date(conn, da)
        # Retrieves the dates relative to t
        dates = self.dates(conn, self.event, t)
        self.t0 = dates[t[0]]
        self.t0m1 = dates[(t[0]-1)]
        self.t1 = dates[t[1]]
        self.t2 = dates[t[2]]
        self.t2m1 = dates[(t[2]-1)]
        self.t3 = dates[t[3]]
        self.conn = conn
        self.t = t

    @staticmethod
    def trade_date(conn, date):
        sql_query0 = \
            "SELECT ff.date " \
            "FROM ff " \
            "WHERE ff.date>=%s "\
            "ORDER by ff.date ASC " \
            "LIMIT 1"
        return conn.fetch_dict(sql_query0, (date,))[0]['date']

    @staticmethod
    def dates(conn, date, t_array):
        # Day zero is the announcement date = (c1-c0)/c0-1
        # If the announcement date is not a trading day -> first day after announcement
        dates = {0: date}
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
            count = 0
            sql_query2 = \
                "SELECT ff.date "\
                "FROM ff " \
                "WHERE ff.date>%s " \
                "ORDER BY ff.date ASC " \
                "LIMIT %s"
            for i in conn.fetch_dict(sql_query2, (dates[0], max(pos))):
                count += 1
                dates[count] = i['date']
        # If the negative list has values, start looking those values up
        if len(neg) > 0:
            sql_query2 = \
                "SELECT ff.date "\
                "FROM ff " \
                "WHERE ff.date<%s " \
                "ORDER BY ff.date DESC " \
                "LIMIT %s"
            count = 0
            for i in conn.fetch_dict(sql_query2, (dates[0], -min(neg)+1)):
                count -= 1
                dates[count] = i['date']
        return dates

    def primary_stock(self, gvkey, start, end, obs_min):
        sql_query0 = \
            "SELECT iid " \
            "FROM com_compm " \
            "WHERE gvkey=%s AND datadate=%s"
        iids = self.conn.fetch_dict(sql_query0, (gvkey, self.event))
        if len(iids) == 1:
            return iids[0]['iid']

        sql_query1 = \
            "SELECT cshtrd, prccd, prcstd, datadate, iid " \
            "FROM com_compm " \
            "WHERE gvkey=%s AND iid=%s AND datadate BETWEEN %s AND %s" \

        trades = dict()
        max_len = 0
        cur1 = self.conn.cur_buff()
        # Loop through all iid
        for i in iids:
            iid = i['iid']
            cur1.execute(sql_query1, (gvkey, iid, start, end))
            obs = 0
            trade = []
            for x in cur1:
                obs += 1
                # Compustat status 3 -> trade toke place
                if x[2] == 3:
                    trade.append(float(x[0])*float(x[1]))
            max_len = max(max_len, len(trade))
            trades[iid] = trade

        med_obs = 0
        med_high = 0
        med_iid = None
        for iid, value in trades.items():
            obs = len(value)
            med = np.median([value+([0]*(max_len-obs))])
            if med > med_high:
                med_iid = iid
                med_high = med
                med_obs = len(value)

        if not med_iid:
            error = 'Gvkey {0} has no primary share between {1} and {2}'
            raise ValueError(dict(msg=error, iid=error.format(gvkey, start, end)))
        # I did catch Compustat in having a missing entry for a stock
        # So add some freedoms to the check
        elif med_obs < obs_min:
            error = 'Gvkey {0}, iid {1} has {2} obs between {3} and {4}, ' \
                    'required obs is {5}'
            error = error.format(gvkey, med_iid, med_obs, start, end, obs_min)
            raise ValueError(dict(msg=error, iid=2))
        else:
            return med_iid

    def cap_fund(self, gvkey, t, prior=False):
        sql0 = (
            "SELECT "
            " datadate, gvkey, prccq, mkvaltq*1000000 mkvaltq, tic, "
            " cshoq*1000000 cshoq, prccq, DLCQ, DLTTQ, PSTKQ, CHEQ, cik, iid "
            "FROM com_fundq "
            "WHERE "
            " datadate < %s AND gvkey = %s AND indfmt='INDL' AND datafmt='STD' AND "
            " popsrc='D' AND consol='C' ORDER BY datadate DESC LIMIT 1")

        sql1 = (
            "SELECT datadate, gvkey, prccq, mkvaltq*1000000 mkvaltq, tic, "
            "  cshoq*1000000 cshoq, prccq, DLCQ, DLTTQ, PSTKQ, CHEQ, cik, iid "
            "FROM com_fundq "
            "WHERE "
            "  datadate >= %s AND gvkey = %s AND indfmt='INDL' AND datafmt='STD' AND"
            "  popsrc='D' AND "
            "  consol='C'"
            "ORDER BY datadate ASC LIMIT 1")

        if prior:
            sql = sql0
        else:
            sql = sql1
        q_p = self.conn.fetch_one(sql, (t, gvkey))
        if not q_p:
            return None

        # The last date in a quarter can be an none trading date!
        # Lookup last trading date
        sql4 = "SELECT date FROM ff WHERE date <= %s ORDER BY date DESC LIMIT 1"
        q_pt = self.conn.fetch_one(sql4, (q_p['datadate'],))['date']
        if q_p['mkvaltq']:
            mv_p = q_p['mkvaltq']
            return dict(mv=mv_p, trade_date=q_pt)

        mv_p = False
        if q_p['cshoq']:
            if q_p['prccq']:
                mv_p = q_p['prccq']*q_p['cshoq']
            else:
                sql = 'SELECT * FROM com_compm WHERE datadate=%s and gvkey=%s and iid=%s'
                pr = self.conn.fetch_one(sql, (q_pt, gvkey, q_p['iid']))
                if not pr:
                    msg = 'Gvkey {0} not listed on date {1}'.format(gvkey, q_pt)
                    raise ValueError(dict(msg=msg, iid=777))
                mv_p = pr['prccd']*q_p['cshoq']
        return dict(mv=mv_p, trade_date=q_pt)

    def cap_t(self, gvkey, iid, t):
        # Get value of public shares
        # On the q before t
        qp = self.cap_fund(gvkey, t, True)
        if not qp:
            error = 'Unable to get funda prior to deal for gvkey: {0}'.format(gvkey)
            raise ValueError(dict(iid=70, msg=error))
        elif not qp['mv']:
            # Go further one quarter back !
            qp = self.cap_fund(gvkey, qp['trade_date'], True)

        if not qp['mv']:
            error = 'Mkt. cap not avaliable for two quarters: {0}'.format(gvkey)
            raise ValueError(dict(iid=7, msg=error))

        qc = self.cap_fund(gvkey, t)
        # The target may not have a ultimo quarter
        ultimo = True
        if not qc or not qc['mv']:
            ultimo = False
            qc = qp

        if qc['trade_date'] == t:
            qp = qc

        mv_p = qp['mv']
        mv_c = qc['mv']
        q_pt = qp['trade_date']
        q_ct = qc['trade_date']

        # If an a stock is not traded on a particular day, then go 10 days back.
        # If still not traded, use the calculated price by Compustat
        sql0 = (
            "SELECT"
            "  SUM(cshoc*prccd) mvp, count(cshoc) classes,"
            "  count(cshoc)-sum(cshoc/cshoc) mvp_check,cshoc, prccd "
            "FROM com_compm a "
            "INNER JOIN ( "
            "  SELECT max(datadate) datadate, a.gvkey, a.iid FROM com_compm a "
            "  INNER JOIN ( "
            "    SELECT min(prcstd) prcstd, gvkey, iid"
            "    FROM com_compm "
            "    WHERE (com_compm.datadate BETWEEN ( "
            "      SELECT date FROM ("
            "        SELECT date FROM ff WHERE date <= %s ORDER BY date DESC limit 10) a"
            "      ORDER BY date ASC limit 1) AND %s) AND "
            "      gvkey = %s AND prcstd in (3, 4) GROUP BY gvkey, iid) b "
            "  ON a.prcstd = b.prcstd and a.gvkey = b.gvkey and a.iid = b.iid "
            "  WHERE a.datadate <= %s "
            "  GROUP BY a.gvkey, a.iid) b "
            "ON a.datadate = b.datadate AND a.iid = b.iid AND a.gvkey = b.gvkey ")

        mp_p = self.conn.fetch_one(sql0, (q_pt, q_pt, gvkey, q_pt))
        # On the q after t
        mp_c = self.conn.fetch_one(sql0, (q_ct, q_ct, gvkey, q_ct))
        # On t
        sql2 = (
            "SELECT SUM(cshoc*prccd) mvp, count(cshoc) classes, "
            "count(cshoc)-sum(cshoc/cshoc) mvp_check, cshoc, prccd "
            "FROM com_compm a "
            "INNER JOIN ( "
            "  SELECT max(datadate) as datadate, gvkey, iid "
            "  FROM com_compm "
            "  WHERE prcstd = 3 AND datadate <= %s AND gvkey = %s AND "
            "    iid in (SELECT iid FROM com_compm WHERE datadate = %s and gvkey = %s) "
            "  GROUP BY gvkey, iid) b "
            "ON a.datadate = b.datadate and a.gvkey = b.gvkey and a.iid = b.iid "
            "GROUP by a.gvkey ")
        mp_t = self.conn.fetch_one(sql2, (t, gvkey, t, gvkey))

        # Missing value in S&P
        if not mp_t:
            error = 'Unable to get record from COMPM for gvkey: {0} on {1}'
            raise ValueError(dict(iid=700, msg=error.format(gvkey, t)))

        if mp_p['mvp_check'] == 0 and mp_c['mvp_check'] == 0 and mp_t['mvp_check'] == 0:
            mvp_p = mp_p['mvp']
            mvp_c = mp_c['mvp']
            classes = mp_t['classes']
            mvp_t = mp_t['mvp']
        else:
            # Very rare
            error = 'Unable to get cap from secd {0}'.format(gvkey)
            raise ValueError(dict(iid=70, msg=error))

        r_p = mvp_p/mv_p
        r_c = mvp_c/mv_c
        valid = True
        # Check if difference is within the tolerance
        if abs(r_p-r_c) > 0.05:
            valid = False

        # Check for private shares
        if r_p > 0.98 or r_c > 0.98:
            r_t = 1
        else:
            r_t = r_p
        mv_t = mvp_t/r_t
        r = dict(mv=mv_t, r=r_t, classes=classes, ultimo=ultimo, valid=valid)
        return r

    def book(self, gvkey, date):
        # This function is an Python implentation of one
        # of the function in the SAS program fama_french_factors_replication.sas
        # (Luis Palacios (WRDS), Premal Vora (Penn State))
        # https://wrds-web.wharton.upenn.edu/wrds/support/Data/
        # _004Research%20Applications/_020Options/Fama%20French%20Factors.cfm
        #
        # at     = Total Assets
        # txditc = Deferred Taxes and Investment Tax Credit
        # ceq    = Common/Ordinary Equity - Total
        # pstkl  = Preferred Stock Liquidating Value
        # pstkrv = Preferred Stock Redemption Value
        # pstk   = Preferred/Preference Stock (Capital) - Total
        sql_query0 =\
            "SELECT datadate, txditc, ceq, pstkl, pstkrv, pstk " \
            "FROM speciale.com_funda "\
            "WHERE gvkey = %s "\
            "   AND %s >= datadate "\
            "   AND indfmt='INDL' " \
            "   AND datafmt='STD' " \
            "   AND popsrc='D' " \
            "   AND consol='C' " \
            "ORDER BY datadate DESC "\
            "LIMIT 1;"

        d = self.conn.fetch_dict(sql_query0, (gvkey, date))
        if len(d) == 0:
            return False
        else:
            d = d[0]
        ceq = d['ceq'] or 0
        ps = d['pstkl'] or d['pstkrv'] or d['pstk'] or 0
        txditc = d['txditc'] or 0
        be = ceq + txditc - ps
        return be*1000000

    def maynes_approach(self, stock_dic, begin, end, t_start):
        # Gvkey, iid, begin, end
        cur0 = self.conn.cur_buff()
        # There can be missing stock data in Compustat.
        # Left joining the FF dataset solves this.

        sql_query0 = \
            "SELECT ff.date, ff.mprem, ff.rf, s.prccd, s.prcstd, s.trfd, s.ajexdi "\
            "FROM ff LEFT JOIN (" \
            "SELECT prccd, prcstd, datadate, trfd, ajexdi " \
            "FROM com_compm " \
            "WHERE gvkey=%s and iid=%s) as s " \
            "ON ff.date = s.datadate "\
            "WHERE ff.date BETWEEN %s and %s "\
            "ORDER BY ff.date ASC"

        s0 = stock_dic['s0']
        cur0.execute(sql_query0, (s0['gvkey'], s0['iid'], begin, end))

        prev_count = None
        prev_close = None
        rm_cache = []
        endogenous = []
        exogenous = []
        mkt = []
        t = []
        #
        count = 0
        for i in cur0:
            count += 1
            status = i[4]
            # market = float(i[1])+float(i[2])
            # ff is in percentage and not decimal
            market = np.log1p((float(i[1])+float(i[2]))/100)
            mkt.append(market)
            # If not status 3, it means that no trade took place that day
            if status == 3:
                adj_close = (float(i[3])*float(i[5]))/float(i[6])
                # A primo is needed before calculations can start
                if prev_count:
                    # The ln of (primo/ultimo)
                    rt = np.log(adj_close/prev_close)

                    # Remember that the FF set is in *100 (pct)
                    rmt = sum(rm_cache)+np.log1p(market)

                    # Add to the lists
                    n = count - prev_count
                    tt = t_start + count - 2
                    t.append(tt)
                    endogenous.append(rt)
                    exogenous.append([n, rmt])

                prev_close = adj_close
                prev_count = count
                rm_cache = []
            elif prev_count:
                rm_cache.append(market)

        return dict(t=t, endogenous=endogenous, exogenous=exogenous, mkt=mkt)

    def maynes_approach_com(self, stock_dic, begin, end, t_start):
        # Gvkey, iid, begin, end
        # There can be missing stock data in Compustat.
        # Left join the FF dataset solves this.
        # Stock_dic = dict(
        #    s0=dict(gvkey='123', iid='bla', cap=123),
        #    s1=dict(gvkey='123', iid='bla', cap=123))

        s0 = stock_dic['s0']
        s1 = stock_dic['s1']

        sql0 = (
            "SELECT "
            "  ff.date, ff.mprem, ff.rf, s0.*, s1.*"
            "FROM ff "
            "LEFT JOIN ("
            "  SELECT datadate datadate0, prccd prccd0, prcstd prcstd0, trfd trfd0,"
            "    ajexdi ajexdi0 "
            "  FROM com_compm "
            "  WHERE gvkey=%s and iid=%s) s0 "
            "ON ff.date = s0.datadate0 "
            "LEFT JOIN ("
            "  SELECT datadate datadate1, prccd prccd1, prcstd prcstd1, trfd trfd1,"
            "    ajexdi ajexdi1 "
            "  FROM com_compm "
            "  WHERE gvkey=%s and iid=%s) s1 "
            "ON ff.date = s1.datadate1 "
            "WHERE ff.date BETWEEN %s and %s "
            "ORDER BY ff.date ASC")

        dic = self.conn.fetch_dict(sql0, (s0['gvkey'], s0['iid'], s1['gvkey'],
                                                s1['iid'], begin, end))
        w0 = float(s0['mv']/(s0['mv']+s1['mv']))
        w1 = 1 - w0

        prev_count = None
        prev_close = None
        rm_cache = []
        endogenous = []
        exogenous = []
        mkt = []
        t = []
        #
        count = 0
        for i in dic:
            count += 1
            # ff is in percentage and not decimal
            market = np.log1p((float(i['mprem'])+float(i['rf']))/100)
            mkt.append(market)
            # If not status 3, it means that no trade took place that day
            if i['prcstd0'] == 3 and i['prcstd1'] == 3:
                adj_close0 = (float(i['prccd0'])*float(i['trfd0']))/float(i['ajexdi0'])
                adj_close1 = (float(i['prccd1'])*float(i['trfd1']))/float(i['ajexdi1'])
                adj_close = adj_close0*w0+adj_close1*w1

                # A primo is needed before calculations can start
                if prev_count:
                    # The ln of (primo/ultimo)
                    rt = np.log(adj_close/prev_close)

                    # Remember that the FF set is in *100 (pct)
                    rmt = sum(rm_cache)+market

                    # Add to the lists
                    n = count - prev_count
                    tt = t_start + count - 2
                    t.append(tt)
                    endogenous.append(rt)
                    exogenous.append([n, rmt])

                prev_close = adj_close
                prev_count = count
                rm_cache = []
            elif prev_count:
                rm_cache.append(market)

        return dict(t=t, endogenous=endogenous, exogenous=exogenous, mkt=mkt)

    def maynes_data(self, stock_dic):
        if len(stock_dic) == 1:
            estimation = self.maynes_approach(stock_dic, self.t0m1, self.t1, self.t[0])
            event = self.maynes_approach(stock_dic, self.t2m1, self.t3, self.t[2])
        else:
            estimation = self.maynes_approach_com(stock_dic, self.t0m1, self.t1,
                                                  self.t[0])
            event = self.maynes_approach_com(stock_dic, self.t2m1, self.t3, self.t[2])
        return dict(estimation=estimation, event=event)


class EventCalc:
    # Takes model_data from the EventData
    def __init__(self, event_data):
        self.data_es = event_data['estimation']
        self.data_ev = event_data['event']
        self.sanity_check_es()
        # self.sanity_check_ev()
        homogen = self.homogen(self.data_es)
        self.params = sm.OLS(homogen['endogenous'], homogen['exogenous']).fit().params
        # Keep this data for test

    def sanity_check_ev(self):
        before = 0
        after = 0
        # Need an abnormal return before and after zero
        for i in self.data_ev['t']:
            if i > 0:
                after += 1
            if i < 0:
                before += 1

        if before < 1 or after < 1:
            error = \
                'Not enough data in event window, obs before time 0: %s, obs after %s'\
                % (before, after)
            raise ValueError(dict(msg=error, iid=4))

    def sanity_check_es(self):
        # At least 50 observation in the estimation window
        if len(self.data_es['t']) < 50:
            error = 'Not enough data in estimation window'
            raise ValueError(dict(msg=error, iid=5))

    def ev_abnormal_returns(self):
        ab = self.abnormal_con(self.params, self.data_ev)
        return ab

    def ev_abnormal_maynes(self):
        data = self.homogen(self.data_ev)
        ab = self.abnormal_con(self.params, data)
        return ab

    def es_abnormal_returns(self):
        ab = self.abnormal_con(self.params, self.data_es)
        return ab

    def es_abnormal_maynes(self):
        data = self.homogen(self.data_es)
        ab = self.abnormal_con(self.params, data)
        return ab

    @staticmethod
    def abnormal_con(params, event_data):
        abnormals = []
        for end, exog in zip(event_data['endogenous'], event_data['exogenous']):
            abnormal = end-sum(np.array(exog)*np.array(params))
            abnormals.append(abnormal)
        return dict(abnormals=abnormals, t=event_data['t'])

    @staticmethod
    def homogen(data):
        endogenous = []
        exogenous = []
        count = 0
        for end, exog in zip(data['endogenous'], data['exogenous']):
            count += 1
            n = exog[0]
            f = 1/np.sqrt(n)
            exogenous.append(f*np.array(exog))
            endogenous.append(f*end)
        return dict(endogenous=endogenous, exogenous=exogenous, t=data['t'])


class EventTest(EventCalc):
    def __init__(self, event_data, t):
        EventCalc.__init__(self, event_data)
        abnormal = self.ev_abnormal_returns()
        self.sum_ab_ev = np.sum(abnormal['abnormals'])
        self.t = t

    def t7_scar(self):
        # Some variables
        l = len(self.data_ev['endogenous'])
        m = len(self.data_es['endogenous'])
        # Calculate the average rm
        sum_exog = np.sum(self.data_es['exogenous'], axis=0)
        rf_avg = sum_exog[1]/sum_exog[0]
        # Calculate variance of abnormals
        abnormal = self.es_abnormal_returns()
        sum_ab_2 = np.sum(np.power(abnormal['abnormals'], 2))
        var_ab = (1/(m-1))*sum_ab_2
        # Calculate the sum of abnormals
        # Get the rm numerator
        avg_mkt = np.average(self.data_es['mkt'])
        nume = np.power(sum(np.array(self.data_ev['mkt'])-avg_mkt), 2)
        deno = sum(np.power(np.array(self.data_es['mkt'])-avg_mkt, 2))
        s_2_car = var_ab*(l+(np.power(l, 2)/m)+(nume/deno))
        s_car = np.sqrt(s_2_car)
        scar = self.sum_ab_ev/s_car
        return scar

    def t4_u(self):
        es_ab = self.es_abnormal_returns()
        ev_ab = self.ev_abnormal_returns()
        al_ab = es_ab['abnormals'] + ev_ab['abnormals']
        ts = es_ab['t'] + ev_ab['t']
        al_ab_ts = []
        # Combine t and ab
        for a, t in zip(al_ab, ts):
            al_ab_ts.append((t, a))
        # Rank after ab
        al_ab_ts.sort(key=lambda x: x[1])
        # Get denominator
        t_1 = len(al_ab_ts)+1
        list_b = []
        count = 1
        # Calculate U
        for i in al_ab_ts:
            list_b.append([i[0], count/t_1])
            count += 1
        # Sort back into t
        list_b.sort(key=lambda x: x[0])
        # Create a list with all the abnormals returns, and zero where no trade
        # Plus as binary list of where a an ab return takes place
        list_c = []
        list_d = []
        # This will fail if the event window does not contain zero,
        # But if the event window does not contain zero, what are we testing for?
        for i in chain(range(self.t[0], self.t[1]+1), range(self.t[2], self.t[3]+1)):
            abb = 0
            b = 0
            if len(list_b) > 0 and list_b[0][0] == i:
                abb = list_b[0][1]
                b = 1
                list_b.pop(0)
            list_c.append(abb)
            list_d.append(b)
        return dict(u=list_c, b=list_d)

    def t2_as(self):
        es_may = self.es_abnormal_maynes()
        ev_may = self.ev_abnormal_maynes()
        T = len(es_may['abnormals'])
        sa = np.sqrt((1/(T-1))*(np.sum(np.power(es_may['abnormals'], 2))))
        aS_it = np.array(ev_may['abnormals'])/sa
        list_a = []
        for i in range(self.t[2], self.t[3]+1):
            abb = 0
            for ab, t in zip(aS_it.tolist(), ev_may['t']):
                if t == i:
                    abb = ab
            list_a.append(abb)
        return list_a


class EventTests:
    def __init__(self, t):
        self.t7 = []
        self.t4 = []
        self.t2 = []
        self.car = []
        self.t = t

    def add_com(self, event_test):
        m = event_test
        self.t7.append(m.t7_scar())
        self.t4.append(m.t4_u())
        self.t2.append(m.t2_as())
        self.car.append(m.sum_ab_ev)

    def caar(self):
        return np.average(self.car)

    def caar_tests(self):
        return [self.caar()] + self.t_all()

    def t7_test(self):
        scar = self.t7
        n = len(scar)
        av_scar = np.average(scar)
        av_s_scar = np.sqrt((1/(n-1))*np.sum((np.power(np.array(scar)-av_scar, 2))))
        bmp = np.sqrt(n)*(av_scar/av_s_scar)
        return bmp

    def t4_test(self):
        u = []
        b = []
        # Start with splitting up into two list
        # b = binary, indicates whether there is return or not
        # u = U from the formula
        for i in self.t4:
            u.append(i['u'])
            b.append(i['b'])
        sum_u = np.sum(np.array(u)-0.5, axis=0)
        sum_n = 1/np.sqrt(np.sum(b, axis=0))
        # Convert any infinite into 0....
        # Only a problem with very small samples
        sum_n[sum_n == np.inf] = 0
        e = (self.t[1]+1-self.t[0])+(self.t[3]+1-self.t[2])
        s_k = np.sqrt((1/e)*sum(np.power(sum_u*sum_n, 2)))
        ev_u = sum_u[self.t[1]+1-self.t[0]:]
        ev_n = sum_n[self.t[1]+1-self.t[0]:]
        t = sum(ev_n*ev_u)/(np.sqrt(len(ev_n))*s_k)
        return t

    def t2_test(self):
        ab = np.sum(self.t2, axis=0)
        a0 = sum(ab)
        win = self.t[3]-self.t[2]+1
        a1 = a0/np.sqrt(len(self.t2)*win)
        return a1

    def t_all(self):
        t_2 = self.t2_test()
        t_4 = self.t4_test()
        t_7 = self.t7_test()
        return [t_2, t_4, t_7, self.normal(t_2), self.normal(t_4), self.normal(t_7)]

    def normal(self, z):
        return np.around(2-st.norm.cdf(abs(z))*2, decimals=4)


class FactorsOth:
    def __init__(self, conn, naics, naics_year, deal_no, year):
        self.naics_year = naics_year
        self.naming = 'usbc'
        self.deal_no = deal_no

        self.conn = conn
        self.year = year
        self.naics = naics
        self.year_census = self.census_year(self.year)
        self.naics_census = self.naics_2007(self.naics, self.year)

    @staticmethod
    def census_year(year):
        years = [1997, 2002, 2007]
        r_year = False
        for i in years:
            if abs(year-i) <= 2:
                r_year = i
        return r_year

    def naics_2007(self, naics, to_year):
        if to_year == 2007:
            return naics

        sql0 = \
            "SELECT to_naics, to_year " \
            "FROM f_cen_ct_naics " \
            "WHERE from_naics = %s " \
            "AND to_year = %s " \
            "AND from_year = %s " \

        # Census does not provide data for converting naics 2007 to naics 1997
        sql2 = (
            "SELECT distinct(b.to_naics) FROM f_cen_ct_naics a "
            "INNER JOIN f_cen_ct_naics b "
            "ON a.from_year = 2007 AND b.from_year = 2002 AND a.to_naics = b.from_naics "
            "WHERE a.from_naics = %s ")

        sql3 = (
            "SELECT acq_naics "
            "FROM deals_naics "
            "WHERE deal_no = %s AND to_year = %s AND from_year = %s")

        # In case the NAICS for that year does not exist, see if we can translate
        # Only if not 2007

        d1 = []
        if to_year == 1997:
            d1 = self.conn.fetch_dict(sql2, (naics,))
        elif to_year == 2002:
            d1 = self.conn.fetch_dict(sql0, (naics, to_year, 2007))

        # Check manual entry in the deals_naics table
        if len(d1) > 1:
            d2 = self.conn.fetch_dict(sql3, (self.deal_no, to_year, 2007))
            if len(d2) == 1:
                d1 = [dict(to_naics=d2[0]['acq_naics'])]

        if len(d1) > 1:
            return False

        try:
            naics = d1[0]['to_naics']
        except IndexError:
            return False
        return naics

    def bea_growth(self):
        b = collections.OrderedDict()
        sql = (
            'SELECT growth '
            'FROM f_gdp '
            'WHERE yyyy = %s')
        dic = self.conn.fetch_one(sql, (self.year,))
        b['bea_growth'] = dic['growth']
        return dict(val=b, desc=None)

    def usbc_con(self):
        b = collections.OrderedDict()
        sql1 = (
            "SELECT * "
            "FROM f_cen_ct "
            "WHERE naics = %s AND yyyy = %s "
            "ORDER BY concenf ASC")

        # Try to fetch using NAICS given
        dic = self.conn.fetch_dict(sql1, (self.naics, self.year_census))

        # If no dice, try the translated
        if not dic and self.naics_census and self.naics_census != self.naics:
            dic = self.conn.fetch_dict(sql1, (self.naics_census, self.year_census))

        if dic:
            # Loop through the output
            # There is a line for each industry concentration, c4, c8, c20, c50
            for i in dic:
                # Btw there is also a first line with some info.
                if i['concenf'] == 1:
                    b['estab'] = i['estab']
                    b['rcptot'] = i['rcptot']
                else:
                    # They are coded like this 804, 820 etc.
                    # Get last two then use int.
                    b['c' + str(int(str(i['concenf'])[-2:]))] = i['pct']
        else:
            b['estab'] = None
            b['rcptot'] = None
            b['c4'] = None
            b['c8'] = None
            b['c20'] = None
            b['c50'] = None
        return dict(val=b, desc=None)

    # USBC/NSF, Concentration
    def usbc_rd(self):
        sql0 = (
            "SELECT * FROM ("
            "SELECT * "
            "FROM f_cen_rd_naics "
            "WHERE bb <= %s AND ff >= %s AND yyyy = %s) as a "
            "INNER JOIN ("
            "  SELECT * FROM f_cen_rd_data"
            "  WHERE total is not null AND net_sales is not null) as b "
            "ON a.field = b.field AND a.yyyy = b.yyyy "
            "ORDER BY a.total_naics ASC "
            "LIMIT 1")

        naics = self.naics
        dic = self.conn.fetch_one(sql0, (naics, naics, self.year_census))
        if not dic and self.naics_census and self.naics_census != self.naics:
            naics = self.naics_census
            dic = self.conn.fetch_one(sql0, (naics, naics, self.year_census))

        b = collections.OrderedDict()
        name = 'nsf_rd'
        desc = dict(nsf_rd='Total R&D expenditures divided by net sales')
        if dic:
            b[name] = (dic['total']/dic['net_sales'])*100
        else:
            b[name] = None
        return dict(val=b, desc=desc)

    def usbc_trade(self):
        sql0 = \
            "SELECT * " \
            "FROM f_cen_ie_man " \
            "WHERE yyyy = %s AND naics = %s"

        dic = self.conn.fetch_one(sql0, (self.year, self.naics))
        if not dic and self.naics_census and self.naics_census != self.naics:
            dic = self.conn.fetch_one(sql0, (self.year, self.naics_census))

        b = collections.OrderedDict()
        if dic:
            b[self.naming + '_export'] = dic['export']
            b[self.naming + '_import'] = dic['import']
        else:
            b[self.naming + '_export'] = None
            b[self.naming + '_import'] = None
        return dict(val=b, desc=None)

    # USBC,  salaray
    def usbc_salary(self):
        if self.year < 1998:
            raise ValueError('Sic code needed for data prior to 1998 (Not implemented)')

        sql0 = "SELECT * " \
               "FROM f_cen_cbp_all " \
               "WHERE year = %s AND indu_code = %s"
        dic = self.conn.fetch_one(sql0, (self.year, self.naics))
        if not dic and self.naics_census and self.naics_census != self.naics:
            dic = self.conn.fetch_one(sql0, (self.year, self.naics_census))

        if dic:
            try:
                sal_avg = dic['ap']/dic['emp']
                sal_rel = dic['rel_pay']
            except ZeroDivisionError:
                sal_avg = None
                sal_rel = None
        else:
            sal_avg = None
            sal_rel = None
        b = collections.OrderedDict()
        name = self.naming + '_sal'
        b[name] = sal_avg

        name = self.naming + '_sal_p'
        b[name] = sal_rel

        return dict(val=b, desc=None)


class FactorsCom:
    def __init__(self, conn, sic, year, min_firms=10):
        self.min_firms = min_firms
        self.sic = sic
        self.year = year
        self.conn = conn

    def com_growth(self):
        b = collections.OrderedDict()
        sql = (
            'SELECT yyyy, (sum(revt)/sum(revt_prev)-1)*100 AS growth '
            'FROM f_com_grow '
            'WHERE yyyy = %s '
            'GROUP by yyyy')
        dic = self.conn.fetch_one(sql, (self.year,))
        b['com_growth'] = dic['growth']
        return dict(val=b, desc=None)

    def com_universal(self, table):
        # f_indu_rev_sum
        sql0 =  \
            "SELECT * FROM {0} "\
            "WHERE yyyy = %s AND firms > %s AND ( "\
            "    (siccd = CAST(SUBSTRING(%s,1,4) as UNSIGNED) AND prec = 4) OR "\
            "    (siccd = CAST(SUBSTRING(%s,1,3) as UNSIGNED) AND prec = 3) OR "\
            "    (siccd = CAST(SUBSTRING(%s,1,2) as UNSIGNED) AND prec = 2) OR "\
            "    (siccd = CAST(SUBSTRING(%s,1,1) as UNSIGNED) AND prec = 1)) "\
            "ORDER BY siccd DESC "\
            "LIMIT 1;".format(table)

        sql1 =  \
            "SELECT * FROM {0} "\
            "WHERE yyyy = %s AND ( "\
            "    (siccd = CAST(SUBSTRING(%s,1,4) as UNSIGNED) AND prec = 4) OR "\
            "    (siccd = CAST(SUBSTRING(%s,1,3) as UNSIGNED) AND prec = 3) OR "\
            "    (siccd = CAST(SUBSTRING(%s,1,2) as UNSIGNED) AND prec = 2) OR "\
            "    (siccd = CAST(SUBSTRING(%s,1,1) as UNSIGNED) AND prec = 1)) "\
            "ORDER BY firms DESC "\
            "LIMIT 1;".format(table)
        sic = self.sic
        sic_row = self.conn.fetch_one(sql0, (self.year, self.min_firms,
                                             sic, sic, sic, sic))
        if not sic_row:
            sic_row = self.conn.fetch_one(sql1, (self.year, sic, sic, sic, sic))
        return sic_row

    def com_factors(self):
        naming = 'com'
        b = collections.OrderedDict()
        com_fac = ('grow', 'assets', 'atrv', 'fxvc', 'xrd')
        com_var = ('equal_w', 'market_w', 'median_w', 'equal_p', 'market_p', 'median_p')

        for fac in com_fac:
            table = 'f_com_' + fac
            dic = self.com_universal(table)
            for var in com_var:
                name = naming + '_' + fac + '_' + var
                b[name] = dic[var]
        return dict(val=b, desc=None)


class MergerData(EventData):
    def __init__(self, conn, tom, d_sett):
        da = tom['date_announced']
        t = d_sett['es_win'] + d_sett['ev_win']
        if d_sett['hoda']:
            t = self.hoda(conn, da, tom['hoda'], d_sett['hoda_pad'], t)
        EventData.__init__(self, conn, da, t)
        self.conn = conn
        tar_mv = self.mv_cache(tom['tar_gvkey'])
        acq_mv = self.mv_cache(tom['acq_gvkey'])

        mv_rr = d_sett['mv_ratio']
        mv_r = tar_mv['mv']/acq_mv['mv']
        if mv_rr and mv_rr > mv_r:
            msg = "Market value ratio is {0} while the required is {1}"
            raise ValueError(dict(iid=0, msg=msg.format(mv_r, mv_rr)))

        acq = dict(acq_mv)
        tar = dict(tar_mv)
        acq_data = self.maynes_data(dict(s0=acq))
        tar_data = self.maynes_data(dict(s0=tar))
        com_data = self.maynes_data(dict(s0=tar, s1=acq))

        self.tom = tom
        self.t = t
        self.acq_cap = acq_mv
        self.tar_cap = tar_mv
        self.deal_sett = d_sett
        self.acq_calc = EventTest(acq_data, t)
        self.tar_calc = EventTest(tar_data, t)
        self.com_calc = EventTest(com_data, t)

    @staticmethod
    def hoda(conn, event, hoda, hoda_pad, t):
        sql_query1 = \
            "SELECT -count(date) c_date " \
            "FROM (" \
            "  SELECT ff.date "\
            "  FROM ff " \
            "  WHERE ff.date<%s AND ff.date>=%s) b " \

        diff = conn.fetch_one(sql_query1, (event, hoda))['c_date']
        ofs = -hoda_pad
        tn2 = ofs + diff
        if tn2 < t[2]:
            diff = tn2 - t[2]
            t[0] = diff + t[0]
            t[1] = diff + t[1]
            t[2] = tn2
        return t

    def mv_cache(self, gvkey):
        date = self.t1
        sql0 = (
            'SELECT * FROM mv_cache WHERE gvkey = %s AND datadate = %s')
        ch = self.conn.fetch_one(sql0, (gvkey, date))
        if ch:
            return ch
        iid = self.primary_stock(gvkey, self.t0m1, self.event, -self.t[0]-2)
        c = self.cap_t(gvkey, iid, date)
        sql1 = (
            'INSERT INTO mv_cache VALUES (%s, %s, %s, %s, %s, %s, %s, %s)')
        self.conn.insert(sql1, (
            date, gvkey, iid, round(c['mv'], 4), round(c['r'], 4), c['classes'],
            c['ultimo'], c['valid']))
        ch = self.conn.fetch_one(sql0, (gvkey, date))
        return ch

    def at_ratios(self, gvkey, date):
        sql0 = "" \
               "SELECT ebitda/at ebitda, ebit/at as ebit FROM com_funda " \
               "WHERE " \
               "  datadate <= %s AND indfmt='INDL' AND datafmt='STD' AND popsrc='D'" \
               "  AND consol='C' AND " \
               "  gvkey = %s ORDER BY datadate DESC LIMIT 1"
        dic = self.conn.fetch_one(sql0, (date, gvkey))
        return dic

    def assets(self, gvkey, date):
        sql0 = "" \
               "SELECT at FROM com_funda " \
               "WHERE " \
               "  datadate <= %s AND indfmt='INDL' AND datafmt='STD' " \
               "  AND popsrc='D' AND consol='C' AND " \
               "  gvkey = %s ORDER BY datadate DESC LIMIT 1"
        dic = self.conn.fetch_dict(sql0, (date, gvkey))[0]
        return dic['at']

    def mebook_info(self):
        # Fin info
        b = collections.OrderedDict()
        d = {}

        code = 'acq_bm'
        d[code] = 'Acquirer P/B'
        b[code] = self.mebook(self.tar_cap['mv'], self.tom['acq_gvkey'])

        code = 'tar_bm'
        d[code] = 'Target P/B'
        b[code] = self.mebook(self.acq_cap['mv'], self.tom['tar_gvkey'])

        return dict(val=b, desc=d)

    def mebook(self, mv, gvkey):
        book = self.book(gvkey, self.t1)
        if not book:
            val = None
        else:
            val = mv/book
        return val

    def ebat_info(self):
        # Fin info
        b = collections.OrderedDict()
        d = {}

        vals = self.at_ratios(self.tom['acq_gvkey'], self.t1)
        code = 'acq_ebat'
        d[code] = 'Acquirer EBITDA/Assets'
        b[code] = vals['ebitda']

        code = 'acq_ebitat'
        d[code] = 'Acquirer EBIT/Assets'
        b[code] = vals['ebit']

        vals = self.at_ratios(self.tom['tar_gvkey'], self.t1)
        code = 'tar_ebat'
        d[code] = 'Target EBITDA/Assets'
        b[code] = vals['ebitda']

        code = 'tar_ebitat'
        d[code] = 'Target EBIT/Assets'
        b[code] = vals['ebit']

        return dict(val=b, desc=d)

    def cash_info(self):
        b = collections.OrderedDict()
        d = {}

        code = 'cash_pct'
        d[code] = 'Cash pct. used to pay for the target'
        val = self.tom['cash_pct']
        b[code] = val

        return dict(val=b, desc=d)

    def basic_info(self):
        b = collections.OrderedDict()
        d = {}

        code = 'deal_no'
        desc = 'SDC deal number'
        val = self.tom['deal_no']
        b[code], d[code] = val, desc

        code = 'date'
        desc = 'Announcement date'
        val = self.tom['date_announced']
        b[code], d[code] = val, desc

        code = 'date_original'
        desc = 'Date original announced (rumour etc.)'
        val = self.tom['date_original']
        b[code], d[code] = val, desc

        code = 'hoda'
        desc = 'hoda'
        val = self.tom['hoda']
        b[code], d[code] = val, desc

        code = 'win'
        desc = 'Primary event window used'
        val = str(self.t)
        b[code], d[code] = val, desc

        code = 'deal_val'
        desc = 'Deal value'
        val= self.tom['deal_val']
        b[code], d[code] = val, desc

        code = 'acq_name'
        desc = 'Acquirer name'
        val= self.tom['acq_name']
        b[code], d[code] = val, desc

        code = 'tar_name'
        desc = 'Target name'
        val= self.tom['tar_name']
        b[code], d[code] = val, desc

        code = 'acq_gvkey'
        desc = 'Acquirer gvkey'
        val= self.tom['acq_gvkey']
        b[code], d[code] = val, desc

        code = 'tar_gvkey'
        desc = 'Target gvkey'
        val = self.tom['tar_gvkey']
        b[code], d[code] = val, desc

        code = 'acq_ev_len'
        desc = 'Acquirer event window length'
        val = len(self.acq_calc.data_ev['t'])
        b[code], d[code] = val, desc

        code = 'tar_ev_len'
        desc = 'Target event window length'
        val = len(self.tar_calc.data_ev['t'])
        b[code], d[code] = val, desc

        code = 'acq_es_len'
        desc = 'Acquirer estimation window length'
        val = len(self.acq_calc.data_es['t'])
        b[code], d[code] = val, desc

        code = 'tar_es_len'
        desc = 'Target estimation window length'
        val = len(self.tar_calc.data_es['t'])
        b[code], d[code] = val, desc

        return dict(val=b, desc=d)

    def stock_info(self):
        # 1.1 Get control status and percentages
        # Transaction/stock info
        b = collections.OrderedDict()
        d = {}

        code = 'acquired'
        d[code] = 'Shares acquired in the transaction'
        b[code] = self.tom['sacq']

        code = 'own'
        d[code] = 'Shares owned prior to the transaction'
        b[code] = self.tom['sown']

        code = 'sought'
        d[code] = 'Shares sought'
        b[code] = self.tom['ssou']

        code = 'control'
        d[code] = 'Whether the acquirer gained control with the company'
        b[code] = self.control()
        return dict(val=b, desc=d)

    def indu_info(self):
        # Industry codes
        # 2.1 Get industry codes from CRSP
        # Indu info
        b = collections.OrderedDict()
        d = {}

        code = 'acq_naics_tom'
        d[code] = 'Acuirer NAICS (Thomson)'
        b[code] = self.tom['acq_naics']

        code = 'acq_naics'
        d[code] = 'Acquirer NAICS (CRSP)'
        b[code] = self.tom['acq_naics_crsp']

        code = 'acq_sic'
        d[code] = 'Acquirer SIC (Thomson)'
        b[code] = self.tom['acq_sic']

        code = 'acq_sic_crsp'
        d[code] = 'Acquirer SIC (CRSP)'
        b[code] = self.tom['acq_sic_crsp']

        code = 'tar_naics'
        d[code] = 'Target NAICS (Thomson)'
        b[code] = self.tom['tar_naics']

        code = 'tar_naic_crsp'
        d[code] = 'Target NAICS (CRSP)'
        b[code] = self.tom['tar_naics_crsp']

        code = 'tar_sic'
        d[code] = 'Target SIC (Thomson)'
        b[code] = self.tom['tar_sic']

        code = 'tar_sic_crsp'
        d[code] = 'Target SIC (CRSP)'
        b[code] = self.tom['tar_sic_crsp']

        code = 'hor_level'
        d[code] = 'Horizontal level'
        b[code] = self.horizontal(self.tom['tar_sic'], self.tom['acq_sic'])
        return dict(val=b, desc=d)

    def fin_indu(self):
        b = collections.OrderedDict()
        d = {}
        code = 'fin_indu'
        d[code] = 'True if target or acquirer industry is financial'
        b[code] = self.fin_com()
        return dict(val=b, desc=d)

    def cap_info(self):
        # Cap info
        b = collections.OrderedDict()
        d = {}
        code = 'acq_mv_tom'
        d[code] = 'Acquirer market value (Thomson)'
        b[code] = self.tom['acq_mar']

        code = 'acq_mv'
        d[code] = 'Acquirer market value'
        b[code] = self.acq_cap['mv']/1000000

        code = 'acq_mv_valid'
        d[code] = 'Acquirer market value valid'
        b[code] = self.acq_cap['valid']

        code = 'acq_factor'
        d[code] = 'Acquirer market factor'
        b[code] = self.acq_cap['r']

        code ='acq_p_classes'
        d[code] = 'Acquirer number of stock classes'
        b[code] = self.acq_cap['classes']

        code = 'tar_mv_tom'
        d[code] = 'Target value (Thomson)'
        b[code] = self.tom['tar_mar']

        code = 'tar_mv'
        d[code] = 'Target value'
        b[code] = self.tar_cap['mv']/1000000

        code = 'tar_mv_valid'
        d[code] = 'Target market value valid'
        b[code] = self.tar_cap['valid']

        code = 'tar_factor'
        d[code] = 'Target market factor'
        b[code] = self.tar_cap['r']

        code = 'tar_p_classes'
        d[code] = 'Target number of stock classes'
        b[code] = self.tar_cap['classes']

        code = 'acq_ultimo'
        d[code] = 'Acquirer market cap is based on a quarter before and after (ultimo)'
        b[code] = self.acq_cap['ultimo']

        code = 'tar_ultimo'
        d[code] = 'Target market cap is based on a quarter before and after (ultimo)'
        b[code] = self.tar_cap['ultimo']
        return dict(val=b, desc=d)

    def comp_info(self):
        # 4. Competition
        b = collections.OrderedDict()
        d = {}

        code = 'acq_acq_oth'
        d[code] = 'Acquirer as an acquirer: ' \
                  'part of another deal during the period (1 if true)'
        val = 0
        if self.comp_sdc(self.tom['acq_gvkey']):
            val = 1
        b[code] = val

        code = 'tar_tar_oth'
        d[code] = 'Target as an target: ' \
                  'part of another deal during the period (1 if true)'
        val = 0
        if self.comp_sdc(self.tom['tar_gvkey'], False):
            val = 1
        b[code] = val

        code = 'tar_tar_oth_tom'
        d[code] = 'Whether Thomson has registered other bids for the target'
        val = 1
        if self.tom['compet'] != 1:
            val = 0
        b[code] = val

        return dict(val=b, desc=d)

    def comp_sdc(self, gvkey, as_acq=True):
        # Test acquirers that have made bids during the stimation period
        sql0 = \
            "SELECT deal_no " \
            "FROM deals " \
            "WHERE {0}=%s AND date_announced BETWEEN %s AND %s"
        if as_acq:
            sql0 = sql0.format('acq_gvkey')
        else:
            sql0 = sql0.format('tar_gvkey')
        cur0 = self.conn.cur_buff()
        cur0.execute(sql0, (gvkey, self.t0m1, self.t3))

        deals = []
        for i in cur0:
            deals.append(i[0])

        if len(deals) > 1:
            return deals
        return False

    def return_info(self):
        # 5. Retrieve abnormal returns
        # CAAR
        b = collections.OrderedDict()
        d = {}

        ev = [self.t[2], self.t[3]]
        acq_ab = CarCalc(self.acq_calc, ev)
        tar_ab = CarCalc(self.tar_calc, ev)
        com_ab = CarCalc(self.com_calc, ev)

        first = True
        for i in [ev] + self.deal_sett['ev_att']:
            if first and self.deal_sett['hoda']:
                sfx = 'dynamic'
                first = False
            else:
                sfx = "{0}_{1}".format(i[0], i[1])
                sfx = re.sub('-', 'n', sfx)

            acq_val = acq_ab.car(i)
            tar_val = tar_ab.car(i)
            com_val = com_ab.car(i)

            code = 'acq_car_' + sfx
            d[code] = 'Acquirer CAR'
            b[code] = acq_val

            code = 'tar_car_' + sfx
            d[code] = 'Target CAR'
            b[code] = tar_val

            code = 'com_car_' + sfx
            d[code] = 'Combined CAR'
            b[code] = com_val

        if not self.deal_sett['hoda'] and 1 == 0:
            ll = range(ev[0], ev[1]+1)
            code = 'acq_ab'
            d[code] = ll
            b[code] = acq_ab.add_blanks()

            code = 'tar_ab'
            d[code] = ll
            b[code] = tar_ab.add_blanks()

            code = 'com_ab'
            d[code] = ll
            b[code] = com_ab.add_blanks()
        return dict(val=b, desc=d)

    @staticmethod
    def horizontal(sic0, sic1):
        a = str(sic0).zfill(4)
        b = str(sic1).zfill(4)
        while len(a) > 0:
            if a == b:
                return len(a)
            a = a[:-1]
            b = b[:-1]
        return 0

    def control(self):
        # 1.1 Get control status and percentages
        acquired = self.tom['sacq']
        own_now = self.tom['sown']

        control = False
        if acquired and own_now:
            own_prev = own_now - acquired
            if own_now > Decimal(50) > own_prev:
                control = True

        return control

    def fin_com(self):
        a = str(self.tom['acq_sic']).zfill(4)
        b = str(self.tom['tar_sic']).zfill(4)
        # Check to see if this is an financial deal
        fin = False
        if a[:1] == '6' or b[:1] == '6':
            fin = True
        return fin


class MergerDataCar:
    def __init__(self, event_calc, ev_win):
        self.ab = event_calc.ev_abnormal_returns()
        self.ev = ev_win

    def add_blanks(self):
        bla = []
        for i in list(range(self.ev[0], self.ev[1]+1)):
            abb = None
            for ab, t in zip(self.ab['abnormals'], self.ab['t']):
                if t == i:
                    abb = ab
            bla.append(abb)
        return bla

    def sum(self, s, e):
        value = 0
        for ab, t in zip(self.ab['abnormals'], self.ab['t']):
            if s <= t <= e:
                value += ab
        if value != 0:
            return value
        else:
            return None

    def car(self, ev_att):
        i = ev_att
        s = i[0]-self.ev[0]
        e = i[1]-i[0]+s+1
        at = self.ab['t'][s:e]
        if len(at) == 0 or \
           (at[0] >= 0 >= i[1] != at[0]) or \
           (at[-1] <= 0 <= i[1] != at[-1]):
            return None
        return self.sum(ev_att[0], ev_att[1])


class MergerItems(FactorsOth, FactorsCom, MergerData):
    def __init__(self, conn, deal, stt_deal):
        # sic = self.match_first(deal['acq_sic'], deal['tar_sic'])
        # naics = self.match_first(deal['acq_naics'], deal['tar_naics'])
        sic = deal['acq_sic']
        naics = deal['acq_naics']
        naics_year = 2007
        year = deal['date_announced'].year
        MergerData.__init__(self, conn, deal, stt_deal)
        FactorsOth.__init__(self, conn, naics, naics_year, deal['deal_no'], year)
        FactorsCom.__init__(self, conn, sic, year)

    def rel_assets(self):
        acq_at = self.assets(self.tom['acq_gvkey'], self.t1)
        tar_at = self.assets(self.tom['tar_gvkey'], self.t1)
        indu_at = self.com_universal('f_com_assets')['median_w']
        if acq_at and tar_at:
            log_acq = np.log(float(acq_at))
            log_com = np.log(float(acq_at+tar_at))
            log_indu = np.log(float(indu_at))
            val_a = abs(log_com/log_indu-1) - abs(log_acq/log_indu-1)
            val_b = log_com/log_indu-1
        else:
            val_a, val_b = None, None

        b = collections.OrderedDict()
        d = {}
        code = 'assets_relprev'
        d[code] = 'Relative assets compared to industry and pre merger asset level'
        b[code] = val_a

        code = 'assets_rel'
        d[code] = 'Relative assets compared to industry'
        b[code] = val_b

        return dict(val=b, desc=d)

    def indu_codes(self):
        b = collections.OrderedDict()
        d = {}
        code = 'com_siccd'
        d[code] = 'Combined siccd'
        b[code] = self.sic

        code = 'com_naics'
        d[code] = 'Combined NAICS'
        b[code] = self.naics
        return dict(val=b, desc=d)

    @staticmethod
    def match_first(in1, in2):
        in1 = str(in1)
        in2 = str(in2)
        ml = max(len(in1), len(in2))
        ln1 = in1.zfill(ml)
        ln2 = in2.zfill(ml)
        while len(ln1) > 1:
            if ln1 == ln2:
                return ln1
            ln1 = ln1[:-1]
            ln2 = ln2[:-1]
        return None


class MergerCsvWriter:
    def __init__(self, settings, conn):
        # Constructs the header in the CSV file
        writer = csv.writer(open(settings['csv_deals'], 'w', newline=''))
        self.writer = writer

        t = settings['deal']['es_win'] + settings['deal']['ev_win']
        self.conn = conn
        self.test = settings['test']

        if self.test:
            if settings['deal']['hoda']:
                print('Test not supported while using hoda')
                exit(1)
            self.tar_test = EventTests(t)
            self.acq_test = EventTests(t)
            self.com_test = EventTests(t)

        self.stt_deal = settings['deal']
        self.stt_items = []
        for k, v in settings['items'].items():
            if v == 'y':
                self.stt_items.append(k)
        self.first = True

    def write_data(self, deal):
        d = MergerItems(self.conn, deal, self.stt_deal)
        item_all = []
        for i in self.stt_items:
            item_all.append(getattr(d, i)())

        # Store data for the tests
        # The only data we need for the entire process
        if self.test:
            self.tar_test.add_com(d.tar_calc)
            self.acq_test.add_com(d.acq_calc)
            self.com_test.add_com(d.com_calc)

        vals = []
        keys = []
        descs = []
        for item in item_all:
            for k, v in item['val'].items():
                if type(v) == list:
                    keys += [k]*len(v)
                    vals += v
                    descs += item['desc'][k]
                else:
                    keys.append(k)
                    vals.append(v)
                    if item['desc']:
                        descs.append(item['desc'][k])
                    else:
                        descs.append(None)

        if self.first:
            self.writer.writerow(keys)
            self.writer.writerow(descs)
            self.first = False
        self.writer.writerow(vals)

    def write_tests(self):
        self.writer.writerow([])
        self.writer.writerow(['', 'CAAR', 't1', 't2', 't3', 't1_p', 't2_p', 't3_p'])
        self.writer.writerow(['acq'] + self.acq_test.caar_tests())
        self.writer.writerow(['tar'] + self.tar_test.caar_tests())
        self.writer.writerow(['com'] + self.com_test.caar_tests())


def loop(settings):
        conn = SqlDB(settings['db'])
        dic = conn.fetch_dict(settings['sql'])
        dt = len(dic)
        writer = MergerCsvWriter(settings, conn)

        error = csv.writer(open(settings['csv_errors'], 'w', newline=''))
        error.writerow(('deal_no', 'date_announced', 'iid', 'msg'))
        count = 0
        for i in dic:
            count += 1
            deal_no = i['deal_no']
            print('%s, Deal no: %s, da: %s'
                  % (round(count/dt, 2), deal_no, i['date_announced']))
            try:
                writer.write_data(i)
            except ValueError as e:
                d = e.args[0]
                if d['iid'] != 0:
                    error.writerow([deal_no, i['date_announced'], d['iid'], d['msg']])

        if settings['test']:
            writer.write_tests()

