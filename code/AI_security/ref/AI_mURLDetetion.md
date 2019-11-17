#
```
AI: Deep Learning for Phishing URL Detection
https://github.com/zpettry/AI-Deep-Learning-for-Phishing-URL-Detection
```
# 20191117
```
要用到flask,
Google colab 難執行
```



### 原始資料的處理技術|白名單與黑名單的處理
```
#!/usr/bin/env python
"""
This file gathers data to be used for pre-processing in training and prediction.
"""
import pandas as pd

def main():

    blacklist = 'phishing_database.csv'
    whitelist = 'whitelist.txt'

    urls = {}
    
    blacklist = pd.read_csv(blacklist)

    #Assign 0 for non-malicious and 1 as malicious for supervised learning.
    for url in blacklist['url']:
        urls[url] = 1
    
    with open(whitelist, 'r') as f:
        lines = f.read().splitlines()
        for url in lines:
            urls[url] = 0

    return urls

if __name__ == "__main__":
    main()
```
