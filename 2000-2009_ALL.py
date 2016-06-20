import returns as r
import os
from collections import OrderedDict
import time
start_time = time.time()

name = '2000-2009_HOA'
sfx = '_[-40,5]'
fd = open(name + '.sql', 'r')
sql0 = fd.read()
fd.close()

settings = dict(
    sql=sql0,
    db='speciale',
    csv_deals=os.path.join(os.getcwd(), 'output', name + '_OUT' + sfx + '.csv'),
    csv_errors=os.path.join(os.getcwd(), 'output', name + '_ERR' + sfx + '.csv'),
    items=OrderedDict([(
        'basic_info', 'y'), (
        'stock_info', 'y'), (
        'cap_info', 'y'), (
        'cash_info', 'y'), (
        'indu_info', 'y'), (
        'fin_indu', 'y'), (
        'comp_info', 'y'), (
        'mebook_info', 'y'), (
        'ebat_info', 'y'), (
        'indu_codes', 'y'), (
        'com_growth', 'y'), (
        'bea_growth', 'y'), (
        'usbc_con', 'y'), (
        'usbc_rd', 'y'), (
        'usbc_salary', 'y'), (
        'usbc_trade', 'y'), (
        'com_factors', 'y'), (
        'rel_assets', 'y'), (
        'return_info', 'y')]),
    deal=dict(
        es_win=[-200-41, -41],
        ev_win=[-40, 5],
        ev_att=[[-35, 5], [-30, 5], [-20, 5], [-10, 5]],
        ev_zero=True,
        hoda=False,
        hoda_pad=5,
        mv_ratio=0.2),
    test=False        # Tests only for ev_win, not supported when using HODA
    )
r.loop(settings)
print("--- %s seconds ---" % (time.time() - start_time))
