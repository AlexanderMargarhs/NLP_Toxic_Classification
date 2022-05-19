# Simple script to check whether Tensorflow is using GPU or CPU
import tensorflow as tf

if __name__ == "__main__":
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	print(sess)
