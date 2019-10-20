
# 1. Install Catboost
```
!pip install catboost
```

# 2. Multiclass Classification
## Wine
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# 導入數據
data = load_wine()

print("data info: ")
print("data contents: ", data.keys()) 
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

print("features: ", data.feature_names)
print("targets: ", data.target_names)
print("data shape: ", data.data.shape)
print("target shape: ", data.target.shape)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2)

print("\nTrain and Testing: ")
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# build model
model = CatBoostClassifier(iterations=5, learning_rate=1, depth=10)

#train
print("\nTraining Model:")
model.fit(X_train, y_train)

# test result
result = model.predict(X_test)

# validation
print("\nTesting result: ")
print("Accuracy: ", accuracy_score(y_test, result))
print("Precision: ", precision_score(y_test, result, average=None))
print("Recall: ", recall_score(y_test, result, average=None))
print("F1_score: ", f1_score(y_test, result, average=None))
print("Precision (average): ", precision_score(y_test, result, average='micro'))
print("Recall (average): ", recall_score(y_test, result, average='micro'))
print("F1_score (average): ", f1_score(y_test, result, average='micro'))
```

## kddcup99
```
from numpy import array
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# 導入數據
data = fetch_kddcup99()

print("data info: ")
print("data contents: ", data.keys()) # dict_keys(['data', 'target', 'DESCR'])
# print("description: ", data.DESCR)
print("data shape: ", data.data.shape)
print("target shape: ", data.target.shape)

proto = {b'tcp' : 1, b'udp' : 2, b'icmp' : 3}

service = {b'aol':1, b'auth':2, b'bgp':3, b'courier':4, b'csnet_ns':5, b'ctf':6, b'daytime':7, b'discard':8, b'domain':9, b'domain_u':10, \
           b'echo':11, b'eco_i':12, b'ecr_i':13, b'efs':14, b'exec':15, b'finger':16, b'ftp':17, b'ftp_data':18, b'gopher':19, b'harvest':20, \
           b'hostnames':21, b'http':22, b'http_2784':23, b'http_443':24, b'http_8001':25, b'imap4':26, b'IRC':27, b'iso_tsap':28, b'klogin':29, b'kshell':30, \
           b'ldap':31, b'link':32, b'login':33, b'mtp':34, b'name':35, b'netbios_dgm':36, b'netbios_ns':37, b'netbios_ssn':38, b'netstat':39, b'nnsp':40, \
           b'nntp':41, b'ntp_u':42, b'other':43, b'pm_dump':44, b'pop_2':45, b'pop_3':46, b'printer':47, b'private':48, b'red_i':49, b'remote_job':50, \
           b'rje':51, b'shell':52, b'smtp':53, b'sql_net':54, b'ssh':55, b'sunrpc':56, b'supdup':57, b'systat':58, b'telnet':59, b'tftp_u':60, \
           b'tim_i':61, b'time':62, b'urh_i':63, b'urp_i':64, b'uucp':65, b'uucp_path':66, b'vmnet':67, b'whois':68, b'X11':69, b'Z39_50':70}

flag = {b'OTH':1, b'REJ':2, b'RSTO':3, b'RSTOS0':4, b'RSTR':5, b'S0':6, b'S1':7, b'S2':8, b'S3':9, b'SF':10, b'SH':11}

label = {b'normal.' : 0, \
         b'ipsweep.' : 1, b'mscan.' : 1, b'nmap.' : 1, b'portsweep.' : 1, b'saint.' : 1, b'satan.' : 1, \
         b'apache2.' : 2, b'back.' : 2, b'land.' : 2, b'mailbomb.' : 2, b'neptune.' : 2, b'pod.' : 2, b'processtable.' : 2, b'smurf.' : 2, b'teardrop.' : 2, b'udpstorm.' : 2, \
         b'buffer_overflow.' : 3, b'httptunnel.' : 3, b'loadmodule.' : 3, b'perl.' : 3, b'ps.' : 3, b'rootkit.' : 3, b'sqlattack.' : 3, b'xterm.' : 3, \
         b'ftp_write.' : 4, b'guess_passwd.' : 4, b'imap.' : 4, b'multihop.' : 4, b'named.' : 4, b'phf.' : 4, b'sendmail.' : 4, b'snmpgetattack.' : 4, \
         b'snmpguess.' : 4, b'spy.' : 4, b'warezclient.' : 4, b'warezmaster.' : 4, b'worm.' : 4, b'xlock.' : 4, b'xsnoop.' : 4}
data_converted = []
target_converted = []

for d in data.data:
  d[1] = proto[d[1]]
  d[2] = service[d[2]]
  d[3] = flag[d[3]]
  data_converted.append(d)

for t in data.target:
  target_converted.append(label[t])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(array(data_converted), array(target_converted), test_size = 0.2)

print("\nTrain and Testing: ")
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# build model
model = CatBoostClassifier(iterations=5, learning_rate=1, depth=10)

#train
print("\nTraining Model:")
model.fit(X_train, y_train)

# test result
result = model.predict(X_test)

# validation
print("\nTesting result: ")
print("Accuracy: ", accuracy_score(y_test, result))
print("Precision: ", precision_score(y_test, result, average=None))
print("Recall: ", recall_score(y_test, result, average=None))
print("F1_score: ", f1_score(y_test, result, average=None))
print("Precision (average): ", precision_score(y_test, result, average='micro'))
print("Recall (average): ", recall_score(y_test, result, average='micro'))
print("F1_score (average): ", f1_score(y_test, result, average='micro'))
```
```
data info: 
data contents:  dict_keys(['data', 'target', 'DESCR'])
data shape:  (494021, 41)
target shape:  (494021,)

Train and Testing: 
X_train shape:  (395216, 41)
X_test shape:  (98805, 41)
y_train shape:  (395216,)
y_test shape:  (98805,)

Training Model:
0:	learn: 0.0460372	total: 1.74s	remaining: 6.95s
1:	learn: 0.0162728	total: 3.29s	remaining: 4.93s
2:	learn: 0.0072137	total: 4.76s	remaining: 3.17s
3:	learn: 0.0043861	total: 6.28s	remaining: 1.57s
4:	learn: 0.0028845	total: 7.74s	remaining: 0us

Testing result: 
Accuracy:  0.9993623804463337
Precision:  [0.99795658 0.98992443 0.99988495 0.8        0.97572816]
Recall:  [0.99923274 0.98496241 0.99987217 0.30769231 0.92201835]
F1_score:  [0.99859425 0.98743719 0.99987856 0.44444444 0.94811321]
Precision (average):  0.9993623804463337
Recall (average):  0.9993623804463337
F1_score (average):  0.9993623804463337
```
