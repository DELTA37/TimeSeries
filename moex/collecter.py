import moex.parser
import matplotlib.pyplot as plt
import os

def getPriceArray(security):
    if not hasattr(getPriceArray, "parser"):
        getPriceArray.parser = moex.parser.Parser({})
    
    engine, market = getPriceArray.parser.getEngineMarketForSecurity(security)
    inf = getPriceArray.parser.iss_history(engine, market, security)[0]

    VALUE                       = inf['VALUE'].fillna(0) 
    OPEN                        = inf['OPEN'].fillna(0)
    LOW                         = inf['LOW'].fillna(0)
    HIGH                        = inf['HIGH'].fillna(0)
    LEGALCLOSEPRICE             = inf['LEGALCLOSEPRICE'].fillna(0)
    WAPRICE                     = inf['WAPRICE'].fillna(0)
    CLOSE                       = inf['CLOSE'].fillna(0)
    VOLUME                      = inf['VOLUME'].fillna(0)
    MARKETPRICE2                = inf['MARKETPRICE2'].fillna(0)
    MARKETPRICE3                = inf['MARKETPRICE3'].fillna(0)
    ADMITTEDQUOTE               = inf['ADMITTEDQUOTE'].fillna(0)
    MP2VALTRD                   = inf['MP2VALTRD'].fillna(0)
    MARKETPRICE3TRADESVALUE     = inf['MARKETPRICE3TRADESVALUE'].fillna(0)
    ADMITTEDVALUE               = inf['ADMITTEDVALUE'].fillna(0)
    TRADEDATE                   = inf['TRADEDATE']

    return {'OPEN' : OPEN, 'HIGH' : HIGH, 'LOW' : LOW, 'CLOSE' : CLOSE, 'DATE' : TRADEDATE}

def PlotData(security, path, splitit = False):
    d = getPriceArray(security)
    if splitit:
        plt.plot(d['OPEN'], label='OPEN', color='b')
        plt.savefig(os.path.join(path, 'OPEN_' + security + '.png'))
        plt.plot(d['CLOSE'], label='CLOSE', color='g')
        plt.savefig(os.path.join(path, 'CLOSE_' + security + '.png'))
        plt.plot(d['HIGH'], label='HIGH', color='r')
        plt.savefig(os.path.join(path, 'HIGH_' + security + '.png'))
        plt.plot(d['LOW'], label='LOW', color='k')
        plt.savefig(os.path.join(path, 'LOW_' + security + '.png'))
    else:
        plt.plot(d['OPEN'], label='OPEN', color='b')
        plt.plot(d['CLOSE'], label='CLOSE', color='g')
        plt.plot(d['HIGH'], label='HIGH', color='r')
        plt.plot(d['LOW'], label='LOW', color='k')
        plt.savefig(os.path.join(path, 'PRICES_' + security + '.png'))
