"""

define graph

restore parameter

run prediction

"""
import tensorflow as tf

class Dector:
    def __init__(self):
        # define graoh
        self.__v1 = tf.get_variable("v1", shape=[3])
        self.__v2 = tf.get_variable("v2", shape=[5])

        saver = tf.train.Saver()
        self.__sess = tf.Session()
        saver.restore(self.__sess, "/Users/xzhangax/PycharmProjects/deploy_test/model/model.ckpt")

    def __del__(self):
        self.__sess.close()

    def is_ambiguous(self,context=None, query=None):
        print("v1 : %s" % self.__sess.run(self.__v1))
        print("v2 : %s" % self.__sess.run(self.__v2))
        return False

if __name__ == '__main__':
    dector = Dector()
    dector.is_ambiguous()