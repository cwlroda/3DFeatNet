from config import BATCH_SIZE
import logging
import tensorflow as tf
# from tensorflow._api.v2 import train
# from tensorflow.python.ops.gen_state_ops import assign_add_eager_fallback
# from tensorflow.python.ops.numpy_ops.np_math_ops import positive

from models.pointnet_common import sample_points, sample_and_group, sample_and_group_all, query_and_group_points
from models.layers import pairwise_dist, MaxPoolAxis, MaxPoolConcat, Conv2D_BN

class Feat3dNet(tf.keras.Model):
    '''
    Implements a fully sequential version of 3DFeatNet, written in idiomatic TensorFlow 2.
    '''
    def __init__(self, train_or_infer, input_shape=None, name="3dFN_seq", param=None):
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

        self.end_points = {}
        
        # Parameters of calculation:
        self._num_clusters = param['num_clusters']
        self._radius = param['BaseScale']
        self._num_samples = param['num_samples']
        self._NoRegress = param['NoRegress']
        self._Attention = param['Attention']

        final_relu = False   # Required parameter for constructing ext_layers

        mlp = [64, 128, 256]
        mlp2 = [128, 64]
        self.logger.info('Detection MLP sizes: {} | {} '.format(mlp, mlp2))

        self.det_layers = []
        for i, num_out_channel in enumerate(mlp):
            self.det_layers.append(
                Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', name="detection_conv_%d" %i)
            )
        self.det_layers.append( MaxPoolAxis(name="det_MaxPoolAxis") ) # Custom layer pooling only on one axis then tiling then concat
        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.det_layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', name="detection_conv_post_%d" %i)
                )

        self.Attention = tf.keras.layers.Conv2D(1, kernel_size=[1,1], strides=[1,1], padding='valid',
                                                activation='softplus', name="detection_attention"
                                               )

        self.Orientation = tf.keras.layers.Conv2D(2, kernel_size=[1,1], strides=[1,1], padding='valid',
                                                    name='detection_orientation', activation=None )

        mlp = [32, 64]
        mlp2 = [128] if self.param['feature_dim'] <= 64 else [256]
        mlp3 = [self.param['feature_dim']]
        self.logger.info('Description MLP sizes: {} | {} | {}'.format(mlp, mlp2, mlp3))
        self.ext_layers = []
        for i, num_out_channel in enumerate(mlp):
            self.ext_layers.append(
                Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', name='description_conv_%d' %i)
            )
        self.ext_layers.append(MaxPoolConcat(name="desc_MaxPoolConcat"))

        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.ext_layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', name='description_conv_mid_%d' %i,
                            activation='relu' if (final_relu or i<len(mlp2) - 1) else None)
                )
        self.ext_layers.append(MaxPoolAxis(name="desc_MaxPoolAxis"))

        if mlp3 is not None:
            for i, num_out_channel in enumerate(mlp3):
                self.ext_layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn=True, padding='valid', name='description_conv_post_%d' %i,
                            activation='relu' if (final_relu or i<len(mlp3) - 1) else None)
                )

    @tf.function
    def call(self, inputs, training=False):
        '''
        Forward pass of network.

        Inputs:
            inputs(tf.Tensor): Input point cloud of dimension (batch_size, ndataset, 3)
            training(bool): indicates training or inference on layers called.
            bypass_detect(tf.Tensor): Defaults to None. If not, it will be substituted into
                the result of the "keypoints" from the detection section, effectively bypassing
                it. This is used in the validation function for training.
        Outputs:
            keypoints
            features
            attention
            end_points
        '''
        inputs = inputs['pointcloud']
        keypoint_subst = inputs['keypoints']
        bypass_detect = inputs['bypass']

        if self.train_or_infer:
            self.end_points['input_pointclouds'] = inputs

        l0_xyz = inputs[:,:,:3]
        # self.logger.info(">>> Shape of input point cloud: ", l0_xyz.shape)
        l0_points = None    # Normal information not used in 3DFeat-Net

        # Compute Sampling and Grouping Ops
        new_xyz = sample_points(l0_xyz, self._num_clusters)
        new_points, idx = query_and_group_points(l0_xyz, l0_points, new_xyz, self._num_samples, 
                            self._radius, knn=False, use_xyz=True, normalize_radius=True, 
                            orientations=None)

        # Compute Conv2d_BN --> MaxPoolAxis --> Conv2d_BN
        for layer in self.det_layers:
            new_points = layer(new_points, training)

        # Compute Attention
        attention = self.Attention(new_points)
        attention = tf.squeeze(attention, axis=[2,3])

        # Compute Orientation
        orientation_xy = self.Orientation(new_points)
        orientation_xy = tf.squeeze(orientation_xy, axis=2)
        orientation_xy = tf.nn.l2_normalize(orientation_xy, axis=2, epsilon=1e-8)
        orientation = tf.atan2(orientation_xy[:, :, 1], orientation_xy[:, :, 0])

        # update end_points
        if self.train_or_infer:
            self.end_points['keypoints'] = new_xyz
            self.end_points['attention'] = attention
            self.end_points['orientation'] = orientation

        # During training, the output of the detector is bypassed.
        orientation_cond = True
        if self._NoRegress:
            orientation_cond = False
        if not self._Attention:
            attention = tf.multiply(attention, 0.0)
            # Set attention==0 (so that a Tensor is always returned)
        
        ### Feature Detection Layer
        if bypass_detect is True:
            new_xyz = keypoint_subst
            attention = tf.multiply(attention, 0.0)
            orientation_cond = False

        ## Feature Extraction Layer
        # Compute Sample and Group. 
        # `npoint` is always 512, so the 'other' branch of PointNetSA is unused.
        new_xyz, new_points, idx, grouped_xyz, end_points_tmp = \
            sample_and_group(npoint=512, radius=self._radius, nsample=self._num_samples, 
                        xyz=l0_xyz, points=l0_points, tnet_spec=None, knn=False, use_xyz=True, 
                        keypoints=new_xyz, orientations=orientation,
                        rotate_orientation=orientation_cond,
                        normalize_radius=True)

        if self.train_or_infer:
            self.end_points.update(end_points_tmp)

        # Compute Conv2d_BN --> MaxPoolConcat --> Conv2d_BN --> MaxPoolAxis --> Conv2d_BN
        for layer in self.ext_layers:
            new_points = layer(new_points, training)

        new_points = tf.squeeze(new_points, [2])
        new_points = tf.nn.l2_normalize(new_points, axis=2, epsilon=1e-8)

        # further computation if Train
        if self.train_or_infer:
            self.end_points['output_xyz'] = new_xyz
            self.end_points['output_features'] = new_points
            
            if training:
                new_xyz = tf.split(new_xyz, 3, axis=0)
                new_points = tf.split(new_points, 3, axis=0)
                # attention = tf.split(attention, 3, axis=0)[0] if attention is not None else None
                attention = tf.split(attention, 3, axis=0)[0]

            return new_xyz, new_points, attention, self.end_points
        else:
            return new_xyz, new_points, attention, {}
    
    # def update_end_points_loss(self, attention_sm, sum_positive, sum_negative):
    #     self.end_points['normalized_attention'] = attention_sm
    #     self.end_points['sum_positive'] = sum_positive
    #     self.end_points['sum_negative'] = sum_negative


class AttentionWeightedAlignmentLoss(tf.keras.losses.Loss):
    '''
    Subclassed Loss function to calculate the Attention Weighted alignment loss.
    '''
    def __init__(self, attention: bool, margin: float, model):
        super().__init__()
        self.attention = attention
        self.margin = margin
        self.model = model

    # @tf.function
    def call(self, y_true, y_pred):
        """ 
        Computes the attention weighted alignment loss as described in our paper.

        Args:
            y_true: Attention from anchor point clouds
            y_pred: List of [anchor_features, positive_features, negative_features]
            model: Feat3dNet object to udpate end_points

        Returns:
            loss (tf.Tensor scalar)
        """
        # Simple stacking
        anchors, positives, negatives = y_pred

        # Computes for each feature of the anchor, the distance to the nearest feature in the positive and negative
        positive_dist = pairwise_dist(anchors, positives)
        best_positive = tf.reduce_min(positive_dist, axis=2)
        # del positive_dist   # hacky memory management

        negative_dist = pairwise_dist(anchors, negatives)
        best_negative = tf.reduce_min(negative_dist, axis=2)
        # del negative_dist   # hacky memory management

        if not self.attention:
            sum_positive = tf.reduce_mean(best_positive, 1)
            sum_negative = tf.reduce_mean(best_negative, 1)
        else:
            attention_sm = y_true / tf.reduce_sum(y_true, axis=1)[:, None]
            sum_positive = tf.reduce_sum(attention_sm * best_positive, 1)
            sum_negative = tf.reduce_sum(attention_sm * best_negative, 1)

            tf.summary.histogram("normalized_attention", attention_sm)
            self.model.end_points['normalized_attention'] = attention_sm

        self.model.end_points['sum_positive'] = sum_positive
        self.model.end_points['sum_negative'] = sum_negative
        triplet_cost = tf.maximum(0., sum_positive - sum_negative + self.margin)
        loss = tf.reduce_mean(triplet_cost)

        return loss

# The rest of these modules were used in development. 
# The 2nd implementation of Feat3dNet may offer more convenient development as it is structured similar to 
# the original version of the Feat3dNet ops. However, it is not tested.
"""
class PointnetSaModule(tf.Module):
    ''' PointNet Set Abstraction (SA) Module. Modified to remove unneeded components (e.g. pooling),
        normalize points based on radius, and for a third layer of MLP
    '''
    
    def __init__(self, mlp, mlp2, mlp3, name="PointnetSaModule", bn=True, final_relu=True):
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
        
        # Define MLP layers based on input dimensions
        for i, num_out_channel in enumerate(mlp):
            self.layers.append(
                Conv2D_BN(num_out_channel, [1,1], bn, padding='valid', name='conv_%d' %i)
            )
        
        self.layers.append(MaxPoolConcat())

        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn, padding='valid', name='conv_mid_%d' %i,
                            activation='relu' if (final_relu or i<len(mlp2) - 1) else None)
                )
        
        self.layers.append(MaxPoolAxis())

        if mlp3 is not None:
            for i, num_out_channel in enumerate(mlp3):
                self.layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn, padding='valid', name='conv_post_%d' %i,
                            activation='relu' if (final_relu or i<len(mlp3) - 1) else None)
                )

    @tf.Module.with_name_scope
    def __call__(self, xyz: tf.Tensor, points: tf.Tensor, npoint: int, radius: float, nsample,
                    is_training, tnet_spec=None, knn=False, use_xyz=True, 
                    keypoints=None, orientations=None, normalize_radius=True):
        '''
        Args:
            xyz (tf.Tensor): (batch_size, ndataset, 3) TF tensor
            points (tf.Tensor): (batch_size, ndataset, num_channel)
            npoint (int32): #points sampled in farthest point sampling
            radius (float): search radius in local region
            nsample (int): Maximum points in each local region
            is_training (bool): calling train or inference
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
        '''

        if npoint is None:
            nsample = xyz.get_shape()[1]   # Number of samples
            new_xyz, new_points, idx, grouped_xyz = \
                sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz, end_points = \
                sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec,
                            knn, use_xyz, keypoints=keypoints,
                            orientations=orientations,
                            normalize_radius=normalize_radius)

        for layer in self.layers:
            new_points = layer(new_points, training=is_training)

        new_points = tf.squeeze(new_points, [2])

        return new_xyz, new_points, idx, end_points

class FeatureDetectionModule(tf.Module):
    '''Detect features in point cloud.
    '''

    def __init__(self, mlp, mlp2, name="FeatureDetectionModule", bn=True):
        '''
        Args:
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP on each region. Set to None or [] to ignore
        name: label
        bn: bool -- Whether to perform batch normalization

        '''
        super(FeatureDetectionModule, self).__init__(name=name)

        self.end_points = {}
        self.layers = []

        # Pre pooling MLP
        for i, num_out_channel in enumerate(mlp):
            self.layers.append(
                Conv2D_BN(num_out_channel, [1,1], bn, padding='valid', name="conv_%d" %i)
            )
        
        # Max pool, then concatenate
        self.layers.append( MaxPoolAxis() ) # Custom layer pooling only on one axis then tiling then concat
        
        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.layers.append(
                    Conv2D_BN(num_out_channel, [1,1], bn, padding='valid', name="conv_post_%d" %i)
                )

        # Two "endpoints" of the calculation
        self.attention = tf.keras.layers.Conv2D(1, kernel_size=[1,1], strides=[1,1], padding='valid',
                                                activation='softplus', name="attention"
                                               )

        self.orientation = tf.keras.layers.Conv2D(2, kernel_size=[1,1], strides=[1,1], padding='valid',
                                                    name='orientation' )
    
    @tf.Module.with_name_scope
    def __call__(self, xyz, points, num_clusters, radius, is_training, num_samples=64):
        '''
        Args:
        xyz (tf.Tensor): Input point cloud of size (batch_size, ndataset, 3)
        points (tf.Tensor): Point features. Unused in 3DFeat-Net
        num_clusters (int): Number of clusters to extract. Set to -1 to use all points
        radius (float): Radius to consider for feature detection
        num_samples: Maximum number of points to consider per cluster
        is_training (bool): train or inference

        Returns:
        new_xyz: Cluster centers
        idx: Indices of points sampled for the clusters
        attention: Output attention weights
        orientation: Output orientation (radians)
        end_points: Unused
        '''

        new_xyz = sample_points(xyz, num_clusters)  # Sample point centers
        new_points, idx = query_and_group_points(xyz, points, new_xyz, num_samples, radius, knn=False, 
                            use_xyz=True, normalize_radius=True, orientations=None)  # Extract clusters

        for layer in self.layers:
            # TODO Check for the shapes' correctness compared to the 3dFeatNet Paper and also the previous model
            # print(">>> Layer", layer.name)
            # print(">>> Shape of input points:", new_points.shape)
            new_points = layer(new_points, training=is_training)
            # print(">>> Shape of output points:", new_points.shape)

        # Attention and orientation regression
        attention_out = self.attention(new_points)
        # print(">>> Attention_out shape:", attention_out.shape)
        attention = tf.squeeze(attention_out, axis=[2, 3])

        orientation_xy = self.orientation(new_points)
        # print(">>> Orientation_xy shape:", orientation_xy.shape)
        orientation_xy = tf.squeeze(orientation_xy, axis=2)
        orientation_xy = tf.nn.l2_normalize(orientation_xy, axis=2, epsilon=1e-8)
        orientation = tf.atan2(orientation_xy[:, :, 1], orientation_xy[:, :, 0])

        return new_xyz, idx, attention, orientation, self.end_points

class FeatureExtractionModule(PointnetSaModule):
    ''' Extract feature descriptors '''

    def __init__(self, mlp, mlp2, mlp3, name="FeatureExtractionModule", bn=True):
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
    
    @tf.Module.with_name_scope
    def __call__(self, l0_xyz: tf.Tensor, l0_points: tf.Tensor, keypoints, orientations, 
                is_training,
                radius=2.0, num_samples=64):
        '''
        Args:
        l0_xyz (tf.Tensor): Input point cloud of size (batch_size, ndataset, 3)
        l0_points (tf.Tensor): Point features. Unused in 3DFeat-Net
        radius: Radius to consider for feature detection
        num_samples: Maximum points in each local region
        keypoints: Keypoints to compute features for
        orientations: Orientation (from detector) to pre-rotate clusters before compute descriptors
        is_training (bool): train or infer

        Returns:
        xyz, features, end_points
        '''
                
        l1_xyz, l1_points, l1_idx, end_points = \
        super(FeatureExtractionModule, self).__call__(
            l0_xyz, l0_points, 512, radius, num_samples, is_training, keypoints=keypoints,
            orientations=orientations, normalize_radius=True
        )

        features = tf.nn.l2_normalize(l1_points, axis=2, epsilon=1e-8)
        # dim is deprecated, use axis instead 
        
        return l1_xyz, features, end_points


class Feat3dNetInference(tf.Module):
    '''
    Contains operations specific to the Inference model of 3DFeatNet.
    Refactored from `Feat3dNet.get_inference_model()`
    '''
    def __init__(self, det_mlps: 'dict', ext_mlps: 'dict', param: 'dict', name, bn=True):
        ''' Constructs the core 3DFeat-Net model.

        Args:
            det_mlps(dict< str, list<int> >): key: 'mlp*' corresponding to the list reqd for the respective mlp
            ext_mlps(dict< str, list<int> >): same as det_mlps
            param (dict): Passes in parameters from outside class for Detection and Extraction
            name (str)
            bn (bool): Whether to perform batch normalization

        Returns:
            xyz, features, attention, end_points

        '''
        super(Feat3dNetInference, self).__init__(name=name)

        self.Detection = FeatureDetectionModule(det_mlps['mlp'], det_mlps['mlp2'], 
                                "Feature_Det_Module", bn)
        self.Extraction = FeatureExtractionModule(ext_mlps['mlp'], ext_mlps['mlp2'], 
                                ext_mlps['mlp3'], "Feature_Ext_Module", bn)

        self._num_clusters = param['num_clusters']
        self._radius = param['BaseScale']
        self._num_samples = param['num_samples']
        self._NoRegress = param['NoRegress']
        self._Attention = param['Attention']

        self.layers = {
            self.Detection.name : self.Detection.layers,
            self.Extraction.name : self.Extraction.layers
        }

        self.end_points = {}
    
    @tf.Module.with_name_scope
    def __call__(self, point_cloud, is_training):
        '''
        Args:
            point_cloud (tf.Tensor): Input point clouds of size (batch_size, ndataset, 3).
            is_training (bool): Training or Inference

        Returns:
            xyz, features, attention, end_points
        '''
        print(">>> >>> Input type to Feat3dNetInfer:", type(point_cloud))
        # Check that the dimension is correct
        l0_xyz = point_cloud[:, :, :3]
        print(">>> Shape of input point cloud: ", l0_xyz.shape)

        l0_points = None    # Normal information not used in 3DFeat-Net

        keypoints, idx, attention, orientation, end_points_temp = \
            self.Detection(l0_xyz, l0_points, self._num_clusters, self._radius, 
                            is_training, self._num_samples)

        self.end_points.update(end_points_temp)
        self.end_points['keypoints'] = keypoints
        self.end_points['attention'] = attention
        self.end_points['orientation'] = orientation

        keypoint_orientation = orientation

        if self._NoRegress:
            keypoint_orientation = None
        if not self._Attention:
            attention = None

        xyz, features, endpoints_temp = \
            self.Extraction(l0_xyz, l0_points, keypoints=keypoints, orientations=keypoint_orientation,
                                    radius=self._radius, is_training=is_training, 
                                    num_samples=self._num_samples)

        self.end_points.update(endpoints_temp)

        return xyz, features, attention, self.end_points


class Feat3dNetTrain(Feat3dNetInference):
    '''
    Contains operations specific to the Inference model of 3DFeatNet.
    Refactored from `Feat3dNet.get_train_model()`, and thus inherits from `Feat3dNetInference`.
    '''
    def __init__(self, det_mlps: 'dict', ext_mlps: 'dict', param, name, bn=True):
        '''
        Constructs the training model. Essentially calls `get_inference_model`, but
        also handles the training triplets.

        Args:
            det_mlps(dict< str, list<int> >): key: 'mlp*' corresponding to the list reqd for the respective mlp
            ext_mlps(dict< str, list<int> >): same as det_mlps
            param (dict): Passes in parameters from outside class for Detection and Extraction
            name (str)
            bn (bool): Whether to perform batch normalization

        '''
        super(Feat3dNetTrain, self).__init__(det_mlps, ext_mlps, param, name, bn)
    
    @tf.Module.with_name_scope
    def __call__(self, point_clouds, is_training):
        '''
        Args:
            point_clouds (tf.Tensor): concatenated list of:
                anchors (tf.Tensor): Anchor point clouds of size (batch_size, ndataset, 3).
                positives (tf.Tensor): Positive point clouds, same size as anchors
                negatives (tf.Tensor): Negative point clouds, same size as anchors
            is_training (bool): Train vs infer
        Returns:
            xyz, features, anchor_attention, end_points
        '''
        print(">>> >>> Input type to Feat3dNetTrain:", type(point_clouds))
        # print(">>> >>>", type(anchors))   # anchors, positives, negatives are EagerTensors.
        # point_clouds = tf.concat([anchors, positives, negatives], axis=0)
        self.end_points['input_pointclouds'] = point_clouds

        xyz, features, attention, endpoints_temp = \
            super(Feat3dNetTrain, self).__call__(point_clouds, is_training)

        self.end_points['output_xyz'] = xyz
        self.end_points['output_features'] = features
        self.end_points.update(endpoints_temp)

        xyz = tf.split(xyz, 3, axis=0)
        features = tf.split(features, 3, axis=0)
        anchor_attention = tf.split(attention, 3, axis=0)[0] if attention is not None else None

        return xyz, features, anchor_attention, self.end_points

class Feat3dNet(tf.keras.Model):
    def __init__(self, train_or_infer, name="3DFeatNet", param=None):
        ''' Constructor: Creates the 3dFeatNet model by calling its relevant sub-objects.

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
        '''
        super(Feat3dNet, self).__init__(name=name)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.param = {}
        self.param.update(param)
        self.logger.info('Model parameters: %s', self.param)

        mlp = [64, 128, 256]
        mlp2 = [128, 64]

        det_mlps = {'mlp': mlp, 'mlp2': mlp2}

        self.logger.info('Detection MLP sizes: {} | {} '.format(mlp, mlp2))

        # Descriptor extraction: Extract descriptors for each cluster
        mlp = [32, 64]
        mlp2 = [128] if self.param['feature_dim'] <= 64 else [256]
        mlp3 = [self.param['feature_dim']]

        ext_mlps = {'mlp': mlp, 'mlp2': mlp2, 'mlp3': mlp3}

        self.logger.info('Description MLP sizes: {} | {} | {}'.format(mlp, mlp2, mlp3))

        self.train_or_infer = train_or_infer
        if train_or_infer: 
            self.Network = Feat3dNetTrain(det_mlps, ext_mlps, self.param, "Feat3dNet")
        else: 
            self.Network = Feat3dNetInference(det_mlps, ext_mlps, self.param, "Feat3dNet")

        # self.layers_ = self.Network.layers

    def call(self, inputs, training=False):
        print(">>> Input to Feat3dNet:", type(inputs))
        return self.Network(inputs, training)
        # Function signature takes in 1 tensor anyway

        '''
        if self.train_or_infer:
            # anchors, positives, negatives = inputs # Inputs is a list of tf.Tensors
            anchors = inputs[0]
            positives = inputs[1]
            negatives = inputs[2]
            print(">>>", type(anchors))

            return self.Network(anchors, positives, negatives, training)
        else:
            point_cloud = inputs[0]
            # point_cloud = inputs['point_cloud']

            return self.Network(point_cloud, training)
        '''

    def feat_3d_net_loss( self, y_true, y_pred ):
        ''' 
        Computes the attention weighted alignment loss as described in our paper.

        Args:
            y_true: Attention from anchor point clouds
            y_pred: List of [anchor_features, positive_features, negative_features]

        Returns:
            loss (tf.Tensor scalar)
        '''
        anchors = y_pred[0]
        positives = y_pred[1]
        negatives = y_pred[2]

        # Computes for each feature of the anchor, the distance to the nearest feature in the positive and negative
        positive_dist = pairwise_dist(anchors, positives)
        negative_dist = pairwise_dist(anchors, negatives)
        best_positive = tf.reduce_min(positive_dist, axis=2)
        best_negative = tf.reduce_min(negative_dist, axis=2)
 
        if not self.param['Attention']:
            sum_positive = tf.reduce_mean(best_positive, 1)
            sum_negative = tf.reduce_mean(best_negative, 1)
        else:
            attention_sm = y_true / tf.reduce_sum(y_true, axis=1)[:, None]
            sum_positive = tf.reduce_sum(attention_sm * best_positive, 1)
            sum_negative = tf.reduce_sum(attention_sm * best_negative, 1)

            # tf.compat.v1.summary.histogram('normalized_attention', attention_sm)
            tf.summary.histogram("normalized_attention", attention_sm)
            self.Network.end_points['normalized_attention'] = attention_sm

        self.Network.end_points['sum_positive'] = sum_positive
        self.Network.end_points['sum_negative'] = sum_negative
        triplet_cost = tf.maximum(0., sum_positive - sum_negative + self.param['margin'])

        loss = tf.reduce_mean(triplet_cost)

        return loss
        # return loss, self.Network.end_points

"""