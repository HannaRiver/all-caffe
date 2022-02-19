import sys
sys.path.insert(0, '/home/hena/caffe-ocr/buildcmake/install/python')
sys.path.insert(0, '/home/hena/tool/protobuf-3.1.0/python')
import caffe
import math
import numpy as np


def SoftMax(net_ans):
    tmp_net = [math.exp(i) for i in net_ans]
    sum_exp = sum(tmp_net)
    return [i/sum_exp for i in tmp_net]

class AggregationCrossEntropyLayer(caffe.Layer):
    """
    Comput the Aggregation Cross Entropy loss for ocr rec plan
    """

    def setup(self, bottom, top):
        print("==============================================================Hi")
        self.dict_size = 1220
        if len(bottom) != 2:
            raise Exception("Need two inputs to computer loss.")

    def reshape(self, bottom, top):
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # top[0].reshape(*bottom[0].data.shape)
    
    def forward(self, bottom, top):
        print("==============================================================Hi1")
        # score = bottom[0].data
        # label = bottom[1].data
        # print(score)
        # print(type(score))
        # print(score.shape)

        # T_ = len(score)
        
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.


    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num

    def get_n_k(self, label):
        pass
        
