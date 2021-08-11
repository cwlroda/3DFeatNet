import logging
import tensorflow as tf
from tensorflow._api.v2 import train

from models.pointnet_common import sample_points, sample_and_group, sample_and_group_all, query_and_group_points
from models.layers_tf2 import pairwise_dist, MaxPoolAxis, MaxPoolConcat

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

    @tf.Module.with_name_scope
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
    
    `compute_det_gradients` is experimental?
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

        self.end_points = {}
        self.end_points['gradients'] = {}
        self.end_points['gradients']['det'] = {}

        self.layers = []
        
        # Define mlp layers based on input dimension
        for i, num_out_channel in enumerate(mlp):
            self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_%d' %(i), activation='relu' )
                              )
            if bn:
                # TODO figure out the appropriate axis. Rest are set to default.
                self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn" ) )

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

        # Two "endpoints" of the calculation
        self.attention = tf.keras.layers.Conv2D(1, kernel_size=[1,1], strides=[1,1], padding='valid',
                                                activation='softplus', name="attention"
                                               )

        self.orientation = tf.keras.layers.Conv2D(2, kernel_size=[1,1], strides=[1,1], padding='valid',
                                                    name='orientation' )
    
    @tf.Module.with_name_scope
    def __call__(self, xyz, points, num_clusters, radius, num_samples=64, compute_det_gradients=True):
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
            
            if layer.name[:4] == "conv" and compute_det_gradients:
                # TODO This might not work out of the box
                self.end_points['gradients']['det']['mlp_{}'.format(last_layer)] = \
                    tf.gradients(new_points, xyz, new_points)

                last_layer += 1
        
        # Attention and orientation regression
        attention_out = self.attention(new_points)
        attention = tf.squeeze(attention_out, axis=[2, 3])

        orientation_xy = self.orientation(new_points)
        orientation_xy = tf.squeeze(orientation_xy, axis=2)
        orientation_xy = tf.nn.l2_normalize(orientation_xy, axis=2, epsilon=1e-8)
        orientation = tf.atan2(orientation_xy[:, :, 1], orientation_xy[:, :, 0])

        return new_xyz, idx, attention, orientation, self.end_points


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
    
    @tf.Module.with_name_scope
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


class Feat3dNetInference(tf.Module):
    '''
    Contains operations specific to the Inference model of 3DFeatNet.
    Refactored from `Feat3dNet.get_inference_model()`
    '''
    def __init__(self, det_mlps: 'dict', ext_mlps: 'dict', params: 'dict', name, is_training, bn=True):
        """ Constructs the core 3DFeat-Net model.

        Args:
            det_mlps(dict< str, list<int> >): key: 'mlp*' corresponding to the list reqd for the respective mlp
            ext_mlps(dict< str, list<int> >): same as det_mlps
            params (dict): Passes in parameters from outside class for Detection and Extraction
            name (str)
            is_training (bool): Set to true only if training, false otherwise
            bn (bool): Whether to perform batch normalization

        Returns:
            xyz, features, attention, end_points

        """
        super(Feat3dNetInference, self).__init__(name=name)

        self.Detection = FeatureDetectionModule(det_mlps['mlp'], det_mlps['mlp2'], 
                                "Feature_Det_Module", bn)
        self.Extraction = FeatureExtractionModule(ext_mlps['mlp'], ext_mlps['mlp2'], 
                                ext_mlps['mlp3'], "Feature_Ext_Module", bn)

        self._num_clusters = params['num_clusters']
        self._radius = params['BaseScale']
        self._num_samples = params['num_samples']
        self._NoRegress = params['NoRegress']
        self._Attention = params['Attention']
    
    @tf.Module.with_name_scope
    def __call__(self, point_cloud, compute_det_gradients=True):
        '''
        Args:
            point_cloud (tf.Tensor): Input point clouds of size (batch_size, ndataset, 3).
            compute_det_gradients (bool): Whether to compute and output gradients for the the Detection stage.

        Returns:
            xyz, features, attention, end_points
        '''

        l0_xyz = point_cloud[:, :, :3]
        l0_points = None    # Normal information not used in 3DFeat-Net
        end_points = {}

        keypoints, idx, attention, orientation, end_points_temp = \
            self.Detection(l0_xyz, l0_points, self._num_clusters, self._radius, 
                            self._num_samples, compute_det_gradients)

        end_points.update(end_points_temp)
        end_points['keypoints'] = keypoints
        end_points['attention'] = attention
        end_points['orientation'] = orientation

        keypoint_orientation = orientation

        if self._NoRegress:
            keypoint_orientation = None
        if not self._Attention:
            attention = None

        xyz, features, endpoints_temp = \
            self.Extraction(l0_xyz, l0_points, keypoints=keypoints,
                                    orientations=keypoint_orientation,
                                    radius=self.param['BaseScale'],
                                    num_samples=self.param['num_samples'])

        end_points.update(endpoints_temp)

        return xyz, features, attention, end_points
    

class Feat3dNetTrain(Feat3dNetInference):
    '''
    Contains operations specific to the Inference model of 3DFeatNet.
    Refactored from `Feat3dNet.get_train_model()`, and thus inherits from `Feat3dNetInference`.
    '''
    def __init__(self, det_mlps: 'dict', ext_mlps: 'dict', params, name, is_training, bn=True):
        '''
        Constructs the training model. Essentially calls `get_inference_model`, but
        also handles the training triplets.

        Args:
            det_mlps(dict< str, list<int> >): key: 'mlp*' corresponding to the list reqd for the respective mlp
            ext_mlps(dict< str, list<int> >): same as det_mlps
            params (dict): Passes in parameters from outside class for Detection and Extraction
            name (str)
            is_training (bool): Set to true only if training, false otherwise
            bn (bool): Whether to perform batch normalization

        '''
        super(Feat3dNetTrain, self).__init__(det_mlps, ext_mlps, params, name, is_training, bn)

        self._params = params
    
    @tf.Module.with_name_scope
    def __call__(self, anchors, positives, negatives, compute_det_gradients=True):
        '''
        Args:
            anchors (tf.Tensor): Anchor point clouds of size (batch_size, ndataset, 3).
            positives (tf.Tensor): Positive point clouds, same size as anchors
            negatives (tf.Tensor): Negative point clouds, same size as anchors
            compute_det_gradients (bool): Whether to compute and output gradients for the the Detection stage.
        Returns:
            xyz, features, anchor_attention, end_points
        '''
        end_points = {}

        point_clouds = tf.concat([anchors, positives, negatives], axis=0)
        end_points['input_pointclouds'] = point_clouds

        xyz, features, attention, endpoints_temp = \
            super(Feat3dNetTrain, self).__call__(point_clouds, self._params, compute_det_gradients)

        end_points['output_xyz'] = xyz
        end_points['output_features'] = features
        end_points.update(endpoints_temp)

        xyz = tf.split(xyz, 3, axis=0, name="xyz")
        features = tf.split(features, 3, axis=0, name="features")
        anchor_attention = tf.split(attention, 3, axis=0, 
            name="anchor_attention")[0] if attention is not None else None

        return xyz, features, anchor_attention, end_points


class Feat3dNet(tf.keras.Model):
    def __init__(self, train_or_infer, param=None):
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
        super(Feat3dNet, self).__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.param = {}
        self.param.update(param)
        self.logger.info('Model parameters: %s', self.param)

        det_mlps = {'mlp': [64, 128, 256], 'mlp2': [128, 64]}

        # Descriptor extraction: Extract descriptors for each cluster
        mlp = [32, 64]
        mlp2 = [128] if self.param['feature_dim'] <= 64 else [256]
        mlp3 = [self.param['feature_dim']]

        ext_mlps = {'mlp': mlp, 'mlp2': mlp2, 'mlp3': mlp3}

        self.logger.info('Descriptor MLP sizes: {} | {} | {}'.format(mlp, mlp2, mlp3))

        if train_or_infer: 
            self.Network = Feat3dNetTrain(det_mlps, ext_mlps, self.param, "Feat3dNet", train_or_infer)
        else: 
            self.Network = Feat3dNetInference(det_mlps, ext_mlps, self.param, "Feat3dNet", train_or_infer)

    def call(self):
        pass

    def get_placeholders(self, data_dim):
        """
        Gets placeholders for data, for triplet loss based training

        Args:
            data_dim: Dimension of point cloud. May be 3 (XYZ), 4 (XYZI), or 6 (XYZRGB or XYZNxNyNz)
                      However for Feat3D-Net we only use the first 3 values

        Returns:
            (anchor_pl, positive_pl, negative_pl)

        """
        pass

    def get_loss(self):
        """ 
        Computes the attention weighted alignment loss as described in our paper.

        Args:
            xyz: Keypoint coordinates (Unused)
            features: List of [anchor_features, positive_features, negative_features]
            anchor_attention: Attention from anchor point clouds
            end_points: end_points, which will be augmented and returned

        Returns:
            loss, end_points
        """
        
        pass

    def get_train_op(self):
        """ 
        Gets training op
        """

        pass

