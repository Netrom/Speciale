from returns import EventData
from database import dbconn
__author__ = 'Morten'


def test():
    conn = dbconn()
    t = [-250, -51, -5, 5]
    da = '2012-10-01'
    gvkey = 7435
    sic = 73
    gvkey_exceptions = [gvkey, 2888]
    event = EventData(conn, da, t)
    # print(event.sic_gvkey(sic))
    ee = event.be_port_equal(sic, gvkey, gvkey_exceptions)
    # ee = event.ff_stock(137377)
    print(ee)

if __name__ == "__main__":
    test()

