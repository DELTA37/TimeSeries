import moex.parser
import matplotlib.pyplot as plt
import os
import pandas as pd

def getSecurityList():
    if not hasattr(getSecurityList, "parser"):
        getSecurityList.parser = moex.parser.Parser({})

    return getSecurityList.parser.iss_securities()


def getPriceArray(security):
    if not hasattr(getPriceArray, "parser"):
        getPriceArray.parser = moex.parser.Parser({})

    def to_float(x, dtype):
        if str.isdigit(x.replace('.','',1)):
            return eval(dtype)(x)
        else:
            return 0
    engine, market = getPriceArray.parser.getEngineMarketForSecurity(security)
    inf = getPriceArray.parser.iss_history(engine, market, security)[0]

#    VALUE                       = inf['VALUE'].fillna(0)
    inf['OPEN']                 = pd.to_numeric(inf['OPEN'], errors='coerce')
    OPEN                        = inf['OPEN'].fillna(0)

    inf['LOW']                  = pd.to_numeric(inf['LOW'], errors='coerce')
    LOW                         = inf['LOW'].fillna(0)

    inf['HIGH']                 = pd.to_numeric(inf['HIGH'], errors='coerce')
    HIGH                        = inf['HIGH'].fillna(0)
#    LEGALCLOSEPRICE             = inf['LEGALCLOSEPRICE'].fillna(0)
#    WAPRICE                     = inf['WAPRICE'].fillna(0)
    inf['CLOSE']                = pd.to_numeric(inf['CLOSE'], errors='coerce')
    CLOSE                       = inf['CLOSE'].fillna(0)
#    VOLUME                      = inf['VOLUME'].fillna(0)
#    MARKETPRICE2                = inf['MARKETPRICE2'].fillna(0)
#    MARKETPRICE3                = inf['MARKETPRICE3'].fillna(0)
#    ADMITTEDQUOTE               = inf['ADMITTEDQUOTE'].fillna(0)
#    MP2VALTRD                   = inf['MP2VALTRD'].fillna(0)
#    MARKETPRICE3TRADESVALUE     = inf['MARKETPRICE3TRADESVALUE'].fillna(0)
#    ADMITTEDVALUE               = inf['ADMITTEDVALUE'].fillna(0)
    TRADEDATE                   = inf['TRADEDATE']

    return {'OPEN' : OPEN, 'HIGH' : HIGH, 'LOW' : LOW, 'CLOSE' : CLOSE, 'DATE' : TRADEDATE}

def PlotData(security, path, splitit = False):
    d = getPriceArray(security)
    if splitit:
        d['OPEN'].to_csv('1.txt')
        plt.plot(d['OPEN'], label='OPEN', color='b')
        plt.savefig(os.path.join(path, 'OPEN_' + security + '.png'))
        plt.clf()
        plt.plot(d['CLOSE'], label='CLOSE', color='g')
        plt.savefig(os.path.join(path, 'CLOSE_' + security + '.png'))
        plt.clf()
        plt.plot(d['HIGH'], label='HIGH', color='r')
        plt.savefig(os.path.join(path, 'HIGH_' + security + '.png'))
        plt.clf()
        plt.plot(d['LOW'], label='LOW', color='k')
        plt.savefig(os.path.join(path, 'LOW_' + security + '.png'))
        plt.clf()
        plt.legend()
        plt.xlabel('Time period')
        plt.ylabel('Price')
    else:
        plt.plot(d['DATA'], d['OPEN'], label='OPEN', color='b')
        plt.plot(d['DATA'], d['CLOSE'], label='CLOSE', color='g')
        plt.plot(d['DATA'], d['HIGH'], label='HIGH', color='r')
        plt.plot(d['DATA'], d['LOW'], label='LOW', color='k')
        plt.legend()
        plt.xlabel('Time period')
        plt.ylabel('Price')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(path, 'PRICES_' + security + '.png'))
