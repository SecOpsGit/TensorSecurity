# TF2_自動求導Gradient tapes
```
https://zhuanlan.zhihu.com/p/69951925
```


```
tensorflow 提供tf.GradientTape api來實現自動求導功能。
只要在tf.GradientTape()上下文中執行的操作，都會被記錄與“tape”中，
然後tensorflow使用反向自動微分來計算相關操作的梯度。

x = tf.ones((2,2))

# 需要計算梯度的操作
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y,y)
# 計算z關於x的梯度
dz_dx = t.gradient(z, x)
print(dz_dx)


tf.Tensor(
[[8. 8.]
 [8. 8.]], shape=(2, 2), dtype=float32)
也可以輸出對中間變數的導數

# 梯度求導只能每個tape一次
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y,y)

dz_dy = t.gradient(z, y)
print(dz_dy)
tf.Tensor(8.0, shape=(), dtype=float32)

預設情況下GradientTape的資源會在執行tf.GradientTape()後被釋放。
如果想多次計算梯度，需要創建一個持久的GradientTape。

with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

dz_dx = t.gradient(z,x)
print(dz_dx)
dz_dy = t.gradient(z, y)
print(dz_dy)
tf.Tensor(
[[8. 8.]
 [8. 8.]], shape=(2, 2), dtype=float32)
tf.Tensor(8.0, shape=(), dtype=float32)

二、記錄控制流
因為tapes記錄了整個操作，所以即使過程中存在python控制流（如if， while），梯度求導也能正常處理。

def f(x, y):
    output = 1.0
    # 根據y的迴圈
    for i in range(y):
        # 根據每一項進行判斷
        if i> 1 and i<5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
        # 返回梯度
        return t.gradient(out, x)
# x為固定值
x = tf.convert_to_tensor(2.0)

print(grad(x, 6))
print(grad(x, 5))
print(grad(x, 4))
tf.Tensor(12.0, shape=(), dtype=float32)
tf.Tensor(12.0, shape=(), dtype=float32)
tf.Tensor(4.0, shape=(), dtype=float32)

三、高階梯度
GradientTape上下文管理器在計算梯度的同時也會保持梯度，所以GradientTape也可以實現高階梯度計算，

x = tf.Variable(1.0)

with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x * x * x
    dy_dx = t2.gradient(y, x)
    print(dy_dx)
    
d2y_d2x = t1.gradient(dy_dx, x)
print(d2y_d2x)

tf.Tensor(3.0, shape=(), dtype=float32)
tf.Tensor(6.0, shape=(), dtype=float32)
```



```

``
