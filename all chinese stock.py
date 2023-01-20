import streamlit as st
import json
import math
import os
import re
import time
import requests
import xlwt



class GetStockList:

    def getOnePage(self, page=1, per_page=500):
        url = 'https://34.push2.eastmoney.com/api/qt/clist/get'
        response = requests.get(url=url, params={
            'cb': 'jQuery1124016636096097589936_1624800100578',
            'pn': page,
            'pz': per_page,
            'po': '1',
            'np': '1',
            'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
            'fltt': '2',
            'invt': '2',
            'fid': 'f3',
            'fs': 'm:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23',
            'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152',
            '_': int(time.time() * 1000)
        })
        result = str(response.content.decode("utf-8"))
        pattern = re.compile(r'(jQuery1124016636096097589936_1624800100578\(|\);)')
        result = str(pattern.sub('', result))
        result = json.loads(result)
        return result

    def getList(self):
        total = 0
        current_page = 1
        path = './china_stock_list.xlsx'
        if os.path.exists(path):
            os.unlink(path)
        workbook = xlwt.Workbook(encoding="utf-8")
        worksheet = workbook.add_sheet('股票列表')
        # 写入表头
        header = ['序号', '代码', '名称', '最新价', '涨跌幅', '涨跌额', '成交量(手)', '成交额', '振幅', '最高', '最低', '今开',
                 '昨收', '量比', '换手率', '市盈率(动态)', '市净率']
        for k, v in enumerate(header):
            worksheet.write(0, k, label=v)

        row = 1
        per_page = 500
        while current_page <= total or total == 0:
            data = self.getOnePage(current_page, per_page)
            if total == 0:
                total = math.ceil(data['data']['total'] / per_page)
            if data['data'] is None:
                break
            stocks = data['data']['diff']
            # 写入excel
            for item in stocks:
                if type(item['f2']) == str:
                    continue
                values = [
                    row, item['f12'], item['f14'], str(item['f2']), str(item['f3']) + '%', item['f4'],
                    self.format(item['f5']), self.format(item['f6']),
                    str(item['f7']) + '%', item['f15'], item['f16'], item['f17'], item['f18'], item['f10'],
                    str(item['f8']) + '%', item['f9'], item['f23'],
                ]
                for k, v in enumerate(values):
                    worksheet.write(row, k, label=v)
                row += 1
            print("第%d页下载完成" % current_page)
            current_page += 1
        workbook.save(path)

    def format(self, num):
        if type(num) == str:
            return num
        if num > 100000000:
            return str(round(num / 100000000, 2)) + '亿'
        if num > 10000:
            return str(round(num / 10000, 2)) + '万'
        return str(num)


gp = GetStockList()
gp.getList()

# print("采集完成，请查看excel文件")
import pandas as pd
data = pd.read_excel(r'china_stock_list.xlsx')
# print(data)

china_stock_list = data['代码'].values.tolist()
#print(china_stock_list)

t_list = []
for china_stock_list_align in china_stock_list:
    t_list.insert(0,str(china_stock_list_align).zfill(6))
#print(t_list)
data["代码"] = t_list
for i in data["代码"]:
    if str(i).startswith('6'):
        i = str(i) + '.ss'
    elif str(i).startswith('3'):
        i = str(i) + '.sz'
    elif str(i).startswith('0'):
        i = str(i) + '.sz'

    t_list.append(i)

st.write(t_list)
