__author__ = 'Morten'
import urllib.request
from returns import Factors
from database import SqlDB, speciale


def lol():
    a = [1, 2, 4, 2]
    a.remove(3)
    print(a)


def test():
    url_base = "http://www2.census.gov/econ1997/EC/"

    for i in [22, 42, 44, 48, 51, 52, 53, 54, 56, 61, 62, 71, 72, 81]:
        name = "E97{0}S4.zip".format(i)
        url_sub = "sector{0}/".format(i)

        g = urllib.request.urlopen(url_base + url_sub + name)
        with open(name, 'b+w') as f:
            f.write(g.read())


def test500():
    f = Factors(conn)
    growth = f.growth_factors(1310)
    print(growth)


def rd():
    f = Factors(conn)
    ctr = f.rd(1998, 812220)
    print(ctr)

conn = SqlDB(speciale())
rd()
