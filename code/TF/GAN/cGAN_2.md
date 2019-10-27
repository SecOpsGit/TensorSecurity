#
```
TensorFlow实现CGAN
2018-06-17 16:56:01 蓬莱道人

https://blog.csdn.net/MOU_IT/article/details/80719465
```

```
#coding=utf-8
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.gridspec as gridspec
import os
import shutil
from scipy.misc import imsave
 
 
# 定義一個mnist資料集的類
class mnistReader():  
    def __init__(self,mnistPath,onehot=True):  
        self.mnistPath=mnistPath
        self.onehot=onehot  
        self.batch_index=0
        print ('read:',self.mnistPath)
        fo = open(self.mnistPath, 'rb')
        self.train_set,self.valid_set,self.test_set = pickle.load(fo,encoding='bytes')
        fo.close()        
        self.data_label_train=list(zip(self.train_set[0],self.train_set[1]))
        np.random.shuffle(self.data_label_train)               
 
    # 獲取下一個訓練集的batch
    def next_train_batch(self,batch_size=100):
        if self.batch_index < int(len(self.data_label_train)/batch_size):  
            # print ("batch_index:",self.batch_index )
            datum=self.data_label_train[self.batch_index*batch_size:(self.batch_index+1)*batch_size]  
            self.batch_index+=1  
            return self._decode(datum,self.onehot)  
        else:  
            self.batch_index=0  
            np.random.shuffle(self.data_label_train)  
            datum=self.data_label_train[self.batch_index*batch_size:(self.batch_index+1)*batch_size]  
            self.batch_index+=1  
            return self._decode(datum,self.onehot)          
    
    # 獲取測試集的資料
    def test_data(self):
        tdata,tlabel=self.test_set
        data_label_test=list(zip(tdata,tlabel))
        return self._decode(data_label_test,self.onehot)    
    
    # 把一個batch的訓練資料轉換為可以放入模型訓練的資料 
    def _decode(self,datum,onehot):  
        rdata=list()     # batch訓練數據
        rlabel=list()  
        if onehot:  
            for d,l in datum:  
                rdata.append(np.reshape(d,[784]))   # 轉變形狀為：一維向量
                hot=np.zeros(10)    
                hot[int(l)]=1            # label設為100維的one-hot向量
                rlabel.append(hot)  
        else:  
            for d,l in datum:  
                rdata.append(np.reshape(d,[784]))  
                rlabel.append(int(l))  
        return rdata,rlabel  
 
 
img_height = 28  # mnist圖片高度
img_width = 28   # mnist圖片寬度
img_size = img_height * img_width   # mnist圖像總的大小
  
to_train = True  
to_restore = False   
output_path = "C-GAN"  # 保存的檔的路徑
  
max_epoch = 500   # 最大反覆運算次數
  
h1_size = 150     # 第一個隱層的單元數
h2_size = 300     # 第二個隱層的單元數
z_size = 100      # 雜訊向量的維度
y_size=10         # 條件變數的維度
batch_size = 256  # batch塊大小
 
 
# 創建生成模型，輸入為雜訊張量，大小為 :batch_size * 100
def build_generator(z_prior,y):  
 
    inputs = tf.concat(axis=1, values=[z_prior, y])
	  # 第一個隱層層
    w1 = tf.Variable(tf.truncated_normal([z_size+y_size, h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)  
    b1 = tf.Variable(tf.zeros([h1_size]), name="g_b1", dtype=tf.float32)  
    h1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)  
    
    # 第二個隱層
    w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)  
    b2 = tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)  
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)   # 偏置是載入batch的每個元素上，也就是和tensor的每行相加
    
    # 輸出層，輸出一個 batch_size * 784 張量，每個元素值在（-1，1）之間
    w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="g_w3", dtype=tf.float32)  
    b3 = tf.Variable(tf.zeros([img_size]), name="g_b3", dtype=tf.float32)  
    h3 = tf.matmul(h2, w3) + b3  
    x_generate = tf.nn.tanh(h3)         # tanh函數輸出(-1,1)之間的某個值    
    g_params = [w1, b1, w2, b2, w3, b3]  
    return x_generate, g_params  
 
 
# 創建生成模型，輸入為真實圖片和生成的圖片
def build_discriminator(x_data, x_generated,y, keep_prob):      
 
    data_and_y = tf.concat(axis=1, values=[x_data, y])  #維度是[batch_size,784 + 10]
    gen_and_y = tf.concat(axis=1, values=[x_generated, y])  #維度是[batch_size,784 + 10]
    
    # 兩個大小batch_size * 784的張量合併為一個 (batch_size*2) * 784的張量，每個tensor的每個元素是一行
    x_in = tf.concat([data_and_y, gen_and_y], 0)    # 相當於把batch_size擴大為2倍    
   
    # 第一個隱層
    w1 = tf.Variable(tf.truncated_normal([img_size+y_size, h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)  
    b1 = tf.Variable(tf.zeros([h2_size]), name="d_b1", dtype=tf.float32)  
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)  
    
    # 第二個隱層
    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)  
    b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)  
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)  
    
    # 輸出層，輸出一個 (2*batch_size) * 1 的張量
    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)  
    b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)  
    h3 = tf.matmul(h2, w3) + b3  
    
    # 計算原始圖片和生成圖片屬於真實圖片的概率，這裡用sigmod函數來計算概率值，屬於(0,1)之間
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None)) # 大小：batch_size*1  
    '''
	  tf.slice(input_, begin, size, name = None)
	  解釋 ：
		這個函數的作用是從輸入資料input中提取出一塊切片,切片的尺寸是size，切片的開始位置是begin。
		切片的尺寸size表示輸出tensor的資料維度，其中size[i]表示在第i維度上面的元素個數。
    '''
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None)) # 大小：batch_size*1
    d_params = [w1, b1, w2, b2, w3, b3]  
    return y_data, y_generated, d_params
 
 
# 開始訓練GAN
def train():  
 
    mnist=mnistReader(mnistPath="E:/testdata/mnist.pkl") 
  
    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")  
    y = tf.placeholder(tf.float32, shape=[batch_size, y_size],name='y')   
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")  
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")  
    global_step = tf.Variable(0, name="global_step", trainable=False)  
  
    x_generated, g_params = build_generator(z_prior,y)  
    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, y , keep_prob)  
  
    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))   # d_loss大小為 batch_size * 1    
    g_loss = - tf.log(y_generated)                          # g_loss大小為 batch_size * 1
  
    optimizer = tf.train.AdamOptimizer(0.0001)   # 定義優化器，學習率0.0001
  
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)  
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)  
  
    init = tf.global_variables_initializer()   
    # saver = tf.train.Saver()    
    sess = tf.Session()    
    sess.run(init)  
  
    if to_restore:  
        chkpt_fname = tf.train.latest_checkpoint(output_path)  
        saver.restore(sess, chkpt_fname)  
    else:  
        if os.path.exists(output_path):  
            shutil.rmtree(output_path)   # 刪除目錄樹
        os.mkdir(output_path)            # 重新創建目錄樹
  
  	# 生成隨機雜訊
    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)  
    
 
    # --------開始訓練模型---------------------------------------------
    for i in range(sess.run(global_step), max_epoch):  
        for j in range(int(50000/batch_size)):  
            print ("epoch:%s, iter:%s" % (i, j)  )
            x_value, y_label=mnist.next_train_batch(batch_size=batch_size)   # 256 * 784
            x_value=np.array(x_value)
            x_value = 2 * x_value.astype(np.float32) - 1     
            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32) 
                         
            sess.run(d_trainer,feed_dict={x_data: x_value,y:y_label, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})  
            if j % 1 == 0:  
                sess.run(g_trainer,feed_dict={x_data: x_value,y:y_label ,z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})  
        
        y_sample = np.zeros(shape=[batch_size, y_size])
        y_sample[:, 7] = 1  #生成的假的標籤
        
        # 生成一個樣本圖片
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val,y:y_sample})  
        show_result(x_gen_val, os.path.join(output_path, "sample%s.jpg" % i))  
        
        # 再次生成一個樣本圖片
        z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)  
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val,y:y_sample})  
        show_result(x_gen_val, os.path.join(output_path, "random_sample%s.jpg" % i))  
        
        # 每次反覆運算保存模型
        sess.run(tf.assign(global_step, i + 1))
        '''
        tf.assign(A, new_number): 這個函數的功能主要是把A的值變為new_number
        '''  
        # saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)  
 
# 保存生成的圖片結果
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):  
	# 由於生成器生成的tensor的每個元素值位於(-1,1)之間，這裡先變成(0,1)之間的值，並把形狀變為圖元矩陣
    batch_res = 0.5*batch_res.reshape((batch_res.shape[0], img_height, img_width))+0.5
 
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]  
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)  
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)  
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)  
    for i, res in enumerate(batch_res):  
        if i >= grid_size[0] * grid_size[1]:  
            break  
        img = (res) * 255                # 生成器生成的是0-1的值，所以要乘以255變成圖元值
        img = img.astype(np.uint8)  
        row = (i // grid_size[0]) * (img_h + grid_pad)  
        col = (i % grid_size[1]) * (img_w + grid_pad)  
        img_grid[row:row + img_h, col:col + img_w] = img  
    imsave(fname, img_grid)  
 
if __name__ == '__main__':
	train()

```
