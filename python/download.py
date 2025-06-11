#下载数据到本地
import requests
from lxml import etree

# 请求地址（周线历史数据）
url = "https://finance.yahoo.com/quote/601006.SS/history/?period1=1156032000&period2=1747814607&frequency=1wk"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36"
}

response = requests.get(url, headers=headers)
html_str = response.text  # 解析 HTML 字符串
html = etree.HTML(html_str)

# 提取数据表格的所有行（tr）
rows = html.xpath('//*[@id="nimbus-app"]/section/section/section/article/div[1]/div[3]/table/tbody/tr')

# 打开文件写入数据
with open('601006.csv', 'w', encoding='utf-8') as f:
    # 写表头（你也可以跳过）
    f.write("Date,Open,High,Low,Close,Adj Close,Volume\n")

    for row in rows:
        # 提取每个单元格（td）文本并清理空格
        cols = row.xpath('.//td//text()')
        cols = [c.strip().replace(',', '') for c in cols]  # 去除逗号避免CSV混乱
        if len(cols) == 7:  # 正常一行有7列
            f.write(','.join(cols) + '\n')