import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np

class Parser:
    @staticmethod
    def type_correct(typename):
        if typename == 'string':
            return 'str'
        elif typename == 'date':
            return 'datetime64[ns]'
        else:
            return typename

    @staticmethod
    def obj_correct(obj, typename):
        if typename == 'datetime64[ns]':
            return pd.to_datetime(obj)
        elif typename == 'str':
            return str(obj)
        elif typename == 'int64':
            try:
                return np.int64(obj)
            except:
                return np.nan
        elif typename == 'int32':
            try:
                return np.int32(obj)
            except:
                return np.nan
        else:
            return obj

    def __init__(self, params):
        pass
    
    def xml2pandas(self, doc):
        root = ET.fromstring(doc)
        assert(root.tag == 'document')
        tables = root.getchildren()
        dfs = []
        for table in tables:
            assert(table.tag == 'data')
            name        = table.attrib['id']

            child       = table.getchildren()
            metadata    = child[0]
            assert(metadata.tag == 'metadata')
            rows        = child[1]
            assert(rows.tag == 'rows')
            
            columns     = metadata.getchildren()[0]
            assert(columns.tag == 'columns')
            
            col_types = dict()
            col_names = []
            for col in columns.getchildren():
                col_names.append(col.attrib['name'])
                col_types[col.attrib['name']] = Parser.type_correct(col.attrib['type'])

            df = pd.DataFrame(columns=col_names).astype(col_types)
             
            for row in rows.getchildren():
                row_attr = {}
                for name in col_names:
                    row_attr[name] = Parser.obj_correct(row.attrib[name], col_types[name])
                row_attr = pd.DataFrame(row_attr, index=[0])
                df = df.append(pd.DataFrame(row_attr, columns=col_names), ignore_index=True)
            dfs.append(df)
        return dfs
    def iss_securities(self, security=None):
        '''
        @params     security    : concrete security
        @type       security    : str
        @return                 : sequrities market description
        @rtype                  : pd.DataFrame()
        '''
        if security == None:
            doc = requests.get('https://iss.moex.com/iss/securities.xml').content.decode()
        else:
            doc = requests.get('https://iss.moex.com/iss/securities/' + security + '.xml').content.decode()
        return self.xml2pandas(doc)

    def iss_securities_indices(self, security):
        '''
        @params     security    : concrete security
        @type       security    : str
        @return                 : list of indices which contain this security
        @rtype                  : pd.DataFrame()
        '''
        doc = requests.get('https://iss.moex.com/iss/securities/' + security + '/indices.xml').content.decode()
        return self.xml2pandas(doc)

    def iss_securities_aggregates(self, security):
        '''
        @params     security    : concrete security
        @type       security    : str
        @return                 : aggregates results of exchanges
        @rtype                  : pd.DataFrame()
        '''
        doc = requests.get('https://iss.moex.com/iss/securities/' + security + '/aggregates.xml').content.decode()
        return self.xml2pandas(doc)

    def iss_securities_bondyields(self, security):
        '''
        @params     security    : concrete security
        @type       security    : str
        @return                 : The rates of return for bonds
        @rtype                  : pd.DataFrame()
        '''
        doc = requests.get('https://iss.moex.com/iss/securities/' + security + '/bondyields.xml').content.decode()
        return self.xml2pandas(doc)

    def iss_engines(self, engine=None):
        '''
        @params     engine      : concrete security
        @type       engine      : str

        @return                 : engine description
        @rtype                  : pd.DataFrame()
        '''
        if engine == None:
            doc = requests.get('https://iss.moex.com/iss/engines.xml').content.decode()
        else:
            doc = requests.get('https://iss.moex.com/iss/engines/' + engine + '.xml').content.decode()
        return self.xml2pandas(doc)
    
    def iss_currentprices(self):
        doc = requests.get('https://iss.moex.com/iss/statistics/engines/stock/currentprices').content.decode()
        return self.xml2pandas(doc)
    
    def iss_securitytypes(self):
        doc = requests.get('https://iss.moex.com/iss/securitytypes').content.decode()
        return self.xml2pandas(doc)
    
    def iss_history(self, engine, market, security=None):
        url = 'https://iss.moex.com/iss/'
        url += 'engines/' + engine + '/'
        url += 'markets/' + market + '/'
        if security == None:
            url += 'securities.xml'
        else:
            url += 'securities/' + security + '.xml'
        doc = requests.get(url).content.decode()
        return self.xml2pandas(doc)


    def iss_history_dates(self, engine, market, security):
        url = 'https://iss.moex.com/iss/'
        url += 'engines/' + engine + '/'
        url += 'markets/' + market + '/'
        url += 'securities/' + security + '/'
        url += 'dates.xml'
        doc = requests.get(url).content.decode()
        return self.xml2pandas(doc)


