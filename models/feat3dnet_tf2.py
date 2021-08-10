import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.linalg_ops import norm
from tensorflow.python.ops.nn_impl import moments

from models.pointnet_common import sample_points, sample_and_group, sample_and_group_all, query_and_group_points
from models.layers import conv2d
from models.layers import pairwise_dist
from models.layers_tf2 import PairwiseDist, maxPoolAxis, maxPoolConcat

class PointnetSaModule(tf.Module):
    """ PointNet Set Abstraction (SA) Module. Modified to remove unneeded components (e.g. pooling),
        normalize points based on radius, and for a third layer of MLP

    Args:
        xyz (tf.Tensor): (batch_size, ndataset, 3) TF tensor
        points (tf.Tensor): (batch_size, ndataset, num_channel)
        npoint (int32): #points sampled in farthest point sampling
        radius (float): search radius in local region
        nsample (int): Maximum points in each local region
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP after max pooling concat
        mlp3: list of int32 -- output size for MLP after second max pooling
        is_training (tf.placeholder): Indicate training/validation
        name (str): name scope
        bn (bool): Whether to perform batch normalizaton
        bn_decay: Decay schedule for batch normalization
        tnet_spec: Unused in Feat3D-Net. Set to None
        knn: Unused in Feat3D-Net. Set to False
        use_xyz: Unused in Feat3D-Net. Set to True
        keypoints: If provided, cluster centers will be fixed to these points (npoint will be ignored)
        orientations (tf.Tensor): Containing orientations from the detector
        normalize_radius (bool): Whether to normalize coordinates [True] based on cluster radius.
        final_relu: Whether to use relu as the final activation function

    Returns:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
        idx: (batch_size, npoint, nsample) int32 -- indices for local regions

    """
    
    def __init__(self, xyz: tf.Tensor, points, npoint, radius, nsample, mlp, mlp2, mlp3, is_training, 
                    name, bn=True, bn_decay=None, tnet_spec=None, knn=False, use_xyz=True,
                    keypoints=None, orientations=None, normalize_radius=True, final_relu=True):
        super(PointnetSaModule, self).__init__(name=name)
        # Initialise layers
        self.layers = []
        
        # Define mlp layers based on input dimension
        for i, num_out_channel in enumerate(mlp):
            self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                                        padding="valid", name='conv_%d' %(i) 
                                                      )
                              )
            if bn:
                # TODO figure out the appropriate axis. Rest are set to default.
                self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn" ) )

        # Max pool, then concatenate
        self.layers.append( maxPoolConcat() ) # Custom layer pooling only on one axis then tiling then concat

        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                                        padding="valid", name='conv_mid_%d' %(i) 
                                                      )
                                  )
                if bn:
                    # TODO figure out the appropriate axis. Rest are set to default.
                    self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn2" ) )
        
        # Max pool again
        self.layers.append( maxPoolAxis() )

        if mlp3 is not None:
            for i, num_out_channel in enumerate(mlp3):
                self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                                        padding="valid", name='conv_post_%d' %(i) 
                                                      )
                                  )
                if bn:
                    # TODO figure out the appropriate axis. Rest are set to default.
                    self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn3" ) )


    def __call__(self, xyz: tf.Tensor, points: tf.Tensor, npoint: int, radius: float, nsample,
                    mlp, mlp2, mlp3, is_training, bn=True, bn_decay=None, tnet_spec=None, 
                    knn=False, use_xyz=True, keypoints=None, orientations=None, 
                    normalize_radius=True, final_relu=True):

        if npoint is None:
            self.nsample = xyz.get_shape()[1]   # Number of samples
            self.new_xyz, self.new_points, self.idx, self.grouped_xyz = \
                sample_and_group_all(xyz, points, use_xyz)
        else:
            self.new_xyz, self.new_points, self.idx, self.grouped_xyz, self.end_points = \
                sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec,
                            knn, use_xyz, keypoints=keypoints,
                            orientations=orientations,
                            normalize_radius=normalize_radius)

        for layer in self.layers:
            self.new_points = layer(self.new_points)

        self.new_points = tf.squeeze(self.new_points, [2])

        return self.new_xyz, self.new_points, self.idx, self.end_points

class FeatureDetectionModule(tf.Module):
    def __init__(self):
        super(FeatureDetectionModule, self).__init__()
    
    def __call__(self):
        pass

class FeatureExtractionModule(tf.Module):
    def __init__(self):
        super(FeatureExtractionModule, self).__init__()
    
    def __call__(self):
        pass