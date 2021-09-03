from config import *
import logging
import tensorflow as tf
from models.pointnet_common import sample_points, sample_and_group, \
                                sample_and_group_all, query_and_group_points
from models.layers import pairwise_dist, MaxPoolAxis, MaxPoolConcat, Conv2D_BN

"""
Implements two separate Feat3dNet cases, one with both Feature Detection and Description,
and one with just the Description, as it is difficult to generate if/else branches in
ONNX -> TensorRT code.
"""

class Feat3dNet(tf.keras.Model):
    '''
    Implements a fully sequential version of 3DFeatNet, written in idiomatic TensorFlow 2.
    '''
    def __init__(self, train_or_infer, name="3dFN_Det_Desc", param=None):
        """ Constructor: Creates the 3dFeatNet model by calling its relevant sub-objects.

        Args:
            train_or_infer (bool): Whether to call `get_inference_model` (false) 
                                   or `get_train_model` (true)

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
        super(Feat3dNet, self).__init__(name=name)

        self.train_or_infer = train_or_infer

        self.logger = logging.getLogger(self.__class__.__name__)
        self.param = {}
        self.param.update(param)
        self.logger.info('Model parameters: %s', self.param)
        self.end_points = {}    # For logging

        # Parameters of calculation:
        self._num_clusters = param['num_clusters']
        self._radius = param['BaseScale']
        self._num_samples = param['num_samples']
        self._NoRegress = param['NoRegress']
        self._Attention = param['Attention']

        # Required parameter for constructing ext_layers
        _final_relu = False

        # Construct Detection MLP Layers
        mlp = [64, 128, 256]
        mlp2 = [128, 64]
        self.logger.info('Detection MLP sizes: {} | {} '.format(mlp, mlp2))

        self.det_layers = []
        for i, num_out_channel in enumerate(mlp):
            self.det_layers.append(
                Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', 
                            name="detection_conv_%d" %i)
            )
        self.det_layers.append( MaxPoolAxis(name="det_MaxPoolAxis") ) # Custom layer pooling only on one axis then tiling then concat
        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.det_layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', 
                                name="detection_conv_post_%d" %i)
                )

        self.Attention = tf.keras.layers.Conv2D(1, kernel_size=[1,1], strides=[1,1], 
                                                padding='valid', activation='softplus', 
                                                name="detection_attention"
                                               )

        self.Orientation = tf.keras.layers.Conv2D(2, kernel_size=[1,1], strides=[1,1], 
                                                    padding='valid',
                                                    name='detection_orientation', 
                                                    activation=None 
                                                )

        # Construct Description MLP Layers
        mlp = [32, 64]
        mlp2 = [128] if self.param['feature_dim'] <= 64 else [256]
        mlp3 = [self.param['feature_dim']]
        self.logger.info('Description MLP sizes: {} | {} | {}'.format(mlp, mlp2, mlp3))
        self.ext_layers = []
        for i, num_out_channel in enumerate(mlp):
            self.ext_layers.append(
                Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', 
                            name='description_conv_%d' %i)
            )
        self.ext_layers.append(MaxPoolConcat(name="desc_MaxPoolConcat"))

        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.ext_layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', 
                            name='description_conv_mid_%d' %i,
                            activation='relu' if (_final_relu or i<len(mlp2) - 1) else None)
                )
        self.ext_layers.append(MaxPoolAxis(name="desc_MaxPoolAxis"))

        if mlp3 is not None:
            for i, num_out_channel in enumerate(mlp3):
                self.ext_layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', 
                            name='description_conv_post_%d' %i,
                            activation='relu' if (_final_relu or i<len(mlp3) - 1) else None)
                )

    @tf.function
    def call(self, pointcloud, training=False):
        '''
        ### Forward pass of network.

        #### Inputs:
        - "pointcloud" (tf.Tensor): Input point cloud of dimension 
            (batch_size, ndataset, 3)
        - training(bool): indicates training or inference on layers called.

        #### Outputs:
        - keypoints
        - features
        - attention
        - end_points
        '''
        self.logger.debug("Tracing the function with input types:")
        self.logger.debug("pointcloud: {}".format(type(pointcloud)))

        if self.train_or_infer:
            self.end_points['input_pointclouds'] = pointcloud

        l0_xyz = pointcloud[:,:,:3] # Slice pointcloud to take the x,y,z dims
        l0_points = None    # Normal information not used in 3DFeat-Net

        # Compute Sampling and Grouping Ops
        new_xyz = sample_points(l0_xyz, self._num_clusters)
        new_points, idx = query_and_group_points(l0_xyz, l0_points, new_xyz,
                            self._num_samples, self._radius, 
                            knn=False, use_xyz=False, use_orientations=False, 
                            use_points=False, normalize_radius=True, orientations=None)

        # Compute Conv2d_BN --> MaxPoolAxis --> Conv2d_BN
        for layer in self.det_layers:
            new_points = layer(new_points, training)
            # self.logger.debug("Layer: {}".format(layer.name))
            # self.logger.debug("new_points has shape: {}".format(new_points.shape))

        # Compute Attention
        attention = self.Attention(new_points)
        attention = tf.squeeze(attention, axis=[2,3], name="attention_squeeze")
        # self.logger.debug("attention has shape: {}".format(attention.shape))

        # Compute Orientation
        orientation_xy = self.Orientation(new_points)
        orientation_xy = tf.squeeze(orientation_xy, axis=2, 
            name="orientation_squeeze")
        orientation_xy = tf.nn.l2_normalize(orientation_xy, axis=2, epsilon=1e-8, 
            name="orientation_l2_norm")
        orientation = tf.atan2(orientation_xy[:, :, 1], orientation_xy[:, :, 0], 
            name="orientation_atan2")
        # self.logger.debug("orientation has shape: {}".format(orientation.shape))

        # During training, the output of the detector is bypassed.
        use_orientation = not(self._NoRegress)

        if self._Attention==False:
            # Set attention==0 (so that a Tensor is always returned)
            attention = tf.multiply(attention, 0.0)

        # update end_points
        if self.train_or_infer:
            self.end_points['keypoints'] = new_xyz
            self.end_points['attention'] = attention
            self.end_points['orientation'] = orientation

        new_xyz, new_points, idx, grouped_xyz, end_points_tmp = \
        sample_and_group(npoint=512, radius=self._radius, nsample=self._num_samples, 
                    xyz=l0_xyz, points=l0_points, 
                    use_xyz=False, use_keypoints=True, use_tnet=False, use_points=False,
                    tnet_spec=None, knn=False, 
                    keypoints=new_xyz, orientations=orientation,
                    rotate_orientation=use_orientation,
                    normalize_radius=True)

        if self.train_or_infer:
            self.end_points.update(end_points_tmp)

        # Compute Conv2d_BN --> MaxPoolConcat --> Conv2d_BN --> MaxPoolAxis --> Conv2d_BN
        for layer in self.ext_layers:
            new_points = layer(new_points, training)
            # self.logger.debug("Layer: {}".format(layer.name))
            # self.logger.debug("new_points has shape: {}".format(new_points.shape))

        new_points = tf.squeeze(new_points, [2], name="features_squeeze")
        new_points = tf.nn.l2_normalize(new_points, axis=2, epsilon=1e-8, name="features_l2_norm")

        # further computation if model is in Train mode
        if self.train_or_infer:
            self.end_points['output_xyz'] = new_xyz
            self.end_points['output_features'] = new_points
            
            if training == True:
                new_xyz = tf.split(new_xyz, 3, axis=0)
                new_points = tf.split(new_points, 3, axis=0)
                attention = tf.split(attention, 3, axis=0)[0]

            return new_xyz, new_points, attention, self.end_points
        else:
            return new_xyz, new_points, attention, {}


class Feat3dNet_Describe(tf.keras.Model):
    '''
    Implements only the Feature Description aspect of `3DFeatNet`.
    DO NOT call this model during training. It is meant to take the weights restored from
    the main 3DFeatNet model.
    '''
    def __init__(self, name="3dFN_Desc", param=None):
        """ Constructor: Creates the 3dFeatNet model by calling its relevant sub-objects.

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
        super(Feat3dNet_Describe, self).__init__(name=name)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.param = {}
        self.param.update(param)
        self.logger.info('Model parameters: %s', self.param)

        # Parameters of calculation:
        self._num_clusters = param['num_clusters']
        self._radius = param['BaseScale']
        self._num_samples = param['num_samples']
        self._NoRegress = param['NoRegress']
        self._Attention = param['Attention']

        # Required parameter for constructing ext_layers
        _final_relu = False

        # Construct Description MLP Layers
        mlp = [32, 64]
        mlp2 = [128] if self.param['feature_dim'] <= 64 else [256]
        mlp3 = [self.param['feature_dim']]
        self.logger.info('Description MLP sizes: {} | {} | {}'.format(mlp, mlp2, mlp3))
        self.ext_layers = []
        for i, num_out_channel in enumerate(mlp):
            self.ext_layers.append(
                Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', 
                            name='description_conv_%d' %i)
            )
        self.ext_layers.append(MaxPoolConcat(name="desc_MaxPoolConcat"))

        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.ext_layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', 
                            name='description_conv_mid_%d' %i,
                            activation='relu' if (_final_relu or i<len(mlp2) - 1) else None)
                )
        self.ext_layers.append(MaxPoolAxis(name="desc_MaxPoolAxis"))

        if mlp3 is not None:
            for i, num_out_channel in enumerate(mlp3):
                self.ext_layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', 
                            name='description_conv_post_%d' %i,
                            activation='relu' if (_final_relu or i<len(mlp3) - 1) else None)
                )

    @tf.function(input_signature=[{

        'pointcloud': tf.TensorSpec([None, None, 6], dtype=tf.float32, name="pointcloud"),
        'keypoints': tf.TensorSpec([None, None, 3], dtype=tf.float32, name="keypoints")
    }])
    def call(self, inputs):
        '''
        ### Forward pass of network.

        #### Inputs:
        - "pointcloud" (tf.Tensor): Input point cloud of dimension 
            (batch_size, ndataset, 3)
        - "keypoints" (tf.Tensor): Value to be substituted into `new_xyz` to bypass
            the Detctor calculation.

        - training(bool): indicates training or inference on layers called.

        #### Outputs:
        - keypoints
        - features
        - attention
        - end_points
        '''
        pointcloud = inputs['pointcloud']
        keypoints = inputs['keypoints']

        self.logger.debug("Tracing the function with input types:")
        self.logger.debug("pointcloud: {}".format(type(pointcloud)))
        self.logger.debug("keypoints: {}".format(type(keypoints)))

        num_clouds = tf.shape(pointcloud)[0]
        num_points = tf.shape(pointcloud)[1]

        l0_xyz = pointcloud[:,:,:3] # Slice pointcloud to take the x,y,z dims
        l0_points = None    # Normal information not used in 3DFeat-Net

        # Attention and Orientation are bypossed
        orientation = tf.zeros([num_clouds, num_points], tf.float32)
        new_xyz = keypoints

        new_xyz, new_points, idx, grouped_xyz, end_points_tmp = \
        sample_and_group(npoint=512, radius=self._radius, nsample=self._num_samples, 
                    xyz=l0_xyz, points=l0_points, 
                    use_xyz=False, use_keypoints=True, use_tnet=False, use_points=False,
                    tnet_spec=None, knn=False, 
                    keypoints=new_xyz, orientations=None,
                    rotate_orientation=False,
                    normalize_radius=True)

        # Compute Conv2d_BN --> MaxPoolConcat --> Conv2d_BN --> MaxPoolAxis --> Conv2d_BN
        for layer in self.ext_layers:
            new_points = layer(new_points, False)
            # self.logger.debug("Layer: {}".format(layer.name))
            # self.logger.debug("new_points has shape: {}".format(new_points.shape))

        new_points = tf.squeeze(new_points, [2], name="features_squeeze")
        new_points = tf.nn.l2_normalize(new_points, axis=2, epsilon=1e-8, name="features_l2_norm")

        return new_xyz, new_points