#
```
print(tf.__version__)
```
```
Oct. 1, 2019: TensorFlow 2.0 Stable!
Aug. 24, 2019: TensorFlow 2.0 rc0
Jun. 8, 2019: TensorFlow 2.0 Beta
Mar. 7, 2019: Tensorflow 2.0 Alpha
Jan. 11, 2019: TensorFlow r2.0 preview
Aug. 14, 2018: TensorFlow 2.0 is coming
```
#### 20191019
```

https://github.com/dragen1860/TensorFlow-2.x-Tutorials
```
```
import tensorflow as tf
import timeit


with tf.device('/cpu:0'):
	cpu_a = tf.random.normal([10000, 1000])
	cpu_b = tf.random.normal([1000, 2000])
	print(cpu_a.device, cpu_b.device)

with tf.device('/gpu:0'):
	gpu_a = tf.random.normal([10000, 1000])
	gpu_b = tf.random.normal([1000, 2000])
	print(gpu_a.device, gpu_b.device)

def cpu_run():
	with tf.device('/cpu:0'):
		c = tf.matmul(cpu_a, cpu_b)
	return c 

def gpu_run():
	with tf.device('/gpu:0'):
		c = tf.matmul(gpu_a, gpu_b)
	return c 


# warm up
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('warmup:', cpu_time, gpu_time)
```
