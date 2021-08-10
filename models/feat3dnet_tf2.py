import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.linalg_ops import norm
from tensorflow.python.ops.nn_impl import moments

from models.pointnet_common import sample_points, sample_and_group, sample_and_group_all, query_and_group_points
from models.layers import conv2d
from models.layers import pairwise_dist
from models.layers_tf2 import PairwiseDist, MaxPoolAxis, MaxPoolConcat

class PointnetSaModule(tf.Module):
    """ PointNet Set Abstraction (SA) Module. Modified to remove unneeded components (e.g. pooling),
        normalize points based on radius, and for a third layer of MLP
    """
    
    def __init__(self, mlp, mlp2, mlp3, name, bn=True, final_relu=True):
        '''
        Args:
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP after max pooling concat
            mlp3: list of int32 -- output size for MLP after second max pooling
            is_training (tf.placeholder): Indicate training/validation
            name (str): name scope
            bn (bool): Whether to perform batch normalizaton
            bn_decay: Decay schedule for batch normalization
            final_relu: Whether to use relu as the final activation function
        '''
        super(PointnetSaModule, self).__init__(name=name)
        # Initialise layers
        self.layers = []
        
        # Define mlp layers based on input dimension
        for i, num_out_channel in enumerate(mlp):
            self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_%d' %(i), activation='relu'
                                        )
                              )
            if bn:
                # TODO figure out the appropriate axis. Rest are set to default.
                self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn" ) )

        # Max pool, then concatenate
        self.layers.append( MaxPoolConcat() ) # Custom layer pooling only on one axis then tiling then concat

        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_mid_%d' %(i),
                                        activation=tf.nn.relu if (final_relu or i < len(mlp2) - 1) else None
                                        )
                                  )
                if bn:
                    # TODO figure out the appropriate axis. Rest are set to default.
                    self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn_mid" ) )
        
        # Max pool again
        self.layers.append( MaxPoolAxis() )

        if mlp3 is not None:
            for i, num_out_channel in enumerate(mlp3):
                self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_post_%d' %(i),
                                        activation=tf.nn.relu if (final_relu or i < len(mlp2) - 1) else None
                                        )
                                  )
                if bn:
                    # TODO figure out the appropriate axis. Rest are set to default.
                    self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn_post" ) )

    def __call__(self, xyz: tf.Tensor, points: tf.Tensor, npoint: int, radius: float, nsample,
                    tnet_spec=None, knn=False, use_xyz=True, keypoints=None, orientations=None, 
                    normalize_radius=True):
        """
        Args:
            xyz (tf.Tensor): (batch_size, ndataset, 3) TF tensor
            points (tf.Tensor): (batch_size, ndataset, num_channel)
            npoint (int32): #points sampled in farthest point sampling
            radius (float): search radius in local region
            nsample (int): Maximum points in each local region
            tnet_spec: Unused in Feat3D-Net. Set to None
            knn: Unused in Feat3D-Net. Set to False
            use_xyz: Unused in Feat3D-Net. Set to True
            keypoints: If provided, cluster centers will be fixed to these points (npoint will be ignored)
            orientations (tf.Tensor): Containing orientations from the detector
            normalize_radius (bool): Whether to normalize coordinates [True] based on cluster radius.

        Returns:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
        """

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

# TODO: Check if defining the gradients within this module (translate from tf1 to tf2...) is reqd
class FeatureDetectionModule(tf.Module):
    """Detect features in point cloud.
    
    `compute_det_gradients` is removed since it seems the orientation is removed.
    """

    def __init__(self, mlp, mlp2, name, bn=True):
        """
        Args:
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP on each region. Set to None or [] to ignore
        name: label
        bn: bool -- Whether to perform batch normalization

        """
        super(FeatureDetectionModule, self).__init__(name=name)
        self.layers = []
        
        # Define mlp layers based on input dimension
        for i, num_out_channel in enumerate(mlp):
            self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_%d' %(i), activation='relu' )
                              )
            if bn:
                # TODO figure out the appropriate axis. Rest are set to default.
                self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn" ) )

            # TODO Implement compute_det_gradients

        # Max pool, then concatenate
        self.layers.append( MaxPoolAxis() ) # Custom layer pooling only on one axis then tiling then concat

        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_mid_%d' %(i)
                                    )
                                  )
                if bn:
                    # TODO figure out the appropriate axis. Rest are set to default.
                    self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn_mid" ) )

            # TODO Implement compute_det_gradients
        
        # Two "endpoints" of the calculation
        self.attention = tf.keras.layers.Conv2D(1, kernel_size=[1,1], strides=[1,1], padding='valid',
                                                activation='softplus', name="attention"
                                               )

        self.orientation = tf.keras.layers.Conv2D(2, kernel_size=[1,1], strides=[1,1], padding='valid',
                                                    name='orientation' )
        
    def __call__(self, xyz, points, num_clusters, radius, num_samples=64):
        """
        Args:
        xyz (tf.Tensor): Input point cloud of size (batch_size, ndataset, 3)
        points (tf.Tensor): Point features. Unused in 3DFeat-Net
        num_clusters (int): Number of clusters to extract. Set to -1 to use all points
        radius (float): Radius to consider for feature detection
        num_samples: Maximum number of points to consider per cluster

        Returns:
        new_xyz: Cluster centers
        idx: Indices of points sampled for the clusters
        attention: Output attention weights
        orientation: Output orientation (radians)
        """

        new_xyz = sample_points(xyz, num_clusters)  # Sample point centers
        new_points, idx = query_and_group_points(xyz, points, new_xyz, num_samples, radius, knn=False, 
                            use_xyz=True, normalize_radius=True, orientations=None)  # Extract clusters

        last_layer = 0

        for layer in self.layers:
            new_points = layer(new_points)
        
        # Attention and orientation regression
        attention_out = self.attention(new_points)
        attention = tf.squeeze(attention_out, axis=[2, 3])

        orientation_xy = self.orientation(new_points)
        orientation_xy = tf.squeeze(orientation_xy, axis=2)
        orientation_xy = tf.nn.l2_normalize(orientation_xy, axis=2, epsilon=1e-8)
        orientation = tf.atan2(orientation_xy[:, :, 1], orientation_xy[:, :, 0])

        return new_xyz, idx, attention, orientation


class FeatureExtractionModule(PointnetSaModule):
    """ Extract feature descriptors """

    def __init__(self, mlp, mlp2, mlp3, name="layer1", bn=True):
        '''
        Args:
        is_training (tf.placeholder): Set to True if training, False during evaluation
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP after max pooling concat
        mlp3: list of int32 -- output size for MLP after second max pooling
        '''
        super(FeatureExtractionModule, self).__init__(
            mlp, mlp2, mlp3, name, bn, final_relu=False
        )
    

    def __call__(self, l0_xyz: tf.Tensor, l0_points: tf.Tensor, 
                radius, num_samples, keypoints, orientations):
        '''
        Args:
        l0_xyz (tf.Tensor): Input point cloud of size (batch_size, ndataset, 3)
        l0_points (tf.Tensor): Point features. Unused in 3DFeat-Net
        keypoints: Keypoints to compute features for
        orientations: Orientation (from detector) to pre-rotate clusters before compute descriptors
        radius: Radius to consider for feature detection
        num_samples: Maximum points in each local region
        use_bn: Whether to perform batch normalizaton

        Returns:
        xyz, features, end_points
        '''
                
        l1_xyz, l1_points, l1_idx, end_points = \
        super(FeatureExtractionModule, self).__call__(
            l0_xyz, l0_points, 512, radius, num_samples, keypoints=keypoints,
            orientations=orientations, normalize_radius=True
        )

        features = tf.nn.l2_normalize(l1_points, axis=2, epsilon=1e-8)
        
        return l1_xyz, features, end_points


# TODO figure out the translation required here.
class Feat3dNetInference(tf.Module):
    '''
    Contains operations specific to the Inference model of 3DFeatNet.
    Refactored from `Feat3dNet.get_inference_model()`
    '''
    def __init__(self, ):
        super(Feat3dNetInference, self).__init__()


class Feat3dNetTrain(Feat3dNetInference):
    '''
    Contains operations specific to the Inference model of 3DFeatNet.
    Refactored from `Feat3dNet.get_train_model()`, and thus inherits from `Feat3dNetInference`.
    '''
    def __init__(self, ):
        super(Feat3dNetTrain, self).__init__()

class Feat3dNet:
    def __init__(self, param=None):
        """ Constructor: Sets the parameters for 3DFeat-Net

        Args:
            param:    Python dict containing the algorithm parameters. It should contain the
                        following fields (square brackets denote paper's parameters':
                        'NoRegress': Whether to skip regression of the keypoint orientation.
                                    [False] (i.e. regress)
                        'BaseScale': Cluster radius. [2.0] (as in the paper)
                        'Attention': Whether to predict the attention. [True]
                        'num_clusters': Number of clusters [512]
                        'num_samples': Maximum number of points per cluster [64]
                        'margin': Triplet loss margin [0.2]
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param = {}
        self.param.update(param)
        self.logger.info('Model parameters: %s', self.param)

    def get_train_model(self, anchors, positives, negatives, is_training, use_bn=True):
        """ Constructs the training model. Essentially calls get_inference_model, but
            also handles the training triplets.

        Args:
            anchors (tf.Tensor): Anchor point clouds of size (batch_size, ndataset, 3).
            positives (tf.Tensor): Positive point clouds, same size as anchors
            negatives (tf.Tensor): Negative point clouds, same size as anchors
            is_training (tf.placeholder): Set to true only if training, false otherwise
            use_bn (bool): Whether to use batch normalization [True]

        Returns:
            xyz, features, anchor_attention, end_points

        """
        pass

    def get_inference_model(self, point_cloud, is_training, use_bn=True, compute_det_gradients=True):
        """ Constructs the core 3DFeat-Net model.

        Args:
            point_cloud (tf.Tensor): Input point clouds of size (batch_size, ndataset, 3).
            is_training (tf.placeholder): Set to true only if training, false otherwise
            use_bn (bool): Whether to perform batch normalization

        Returns:
            xyz, features, attention, end_points
        """
        end_points = {}

        l0_xyz = point_cloud[:, :, :3]
        l0_points = None  # Normal information not used in 3DFeat-Net

        # Detection: Sample many clusters and computer attention weights and orientations
        num_clusters = self.param['num_clusters']

        # define a variable scope for this

        return xyz, features, attention, end_points


    def get_loss(self):
        pass

    def get_train_op(self):
        pass

