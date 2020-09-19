from joint_cluster_cnn import *

tf.compat.v1.disable_eager_execution()
# joint_cluster_cnn('usps', RC = True, updateCNN = True, eta = 0.9).run()
joint_cluster_cnn('mnist-test', RC = True, updateCNN = True, eta = 0.9).run()
#joint_cluster_cnn('mnist-full', RC = True, updateCNN = True, eta = 0.9).run()
# joint_cluster_cnn('coil20', RC = True, updateCNN = True, eta = 0.9).run()
# joint_cluster_cnn('coil100', RC = True, updateCNN = True, eta = 0.9).run()
#joint_cluster_cnn('umist', RC = True, updateCNN = True, eta = 0.2).run()
