# Log Tue 27/07/21
- Used TensorRT to accelerate a custom model (TF2 -> ONNX -> TensorRT)
- Attempt to convert the 3DFeatNet model into TensorRT, but ran into issues:
    ```bash
    cd/workspace/TensorRT/quickstart/threeDfeatNet
    # or wherever your directory is
    python tf2trt_converter.py  --format ckpt2trt \
    --meta_path "./ckpt/checkpoint.ckpt-210000.meta" \
    --ckpt_path "./ckpt/checkpoint.ckpt-210000"
    ```
    - When converting directly to TensorRT, complains that TensorRT is not the same version (Docker uses 8.0.x vs 'registered' 7.x.x)
    - Runs into a strange 'eager execution' assertion error, perhaps because Tf1 is expected but Tf2 is used?

- Attempt to convert model into ONNX, but similarly ran into issues:
    - Trying to convert TF checkpoint using
    ```bash
    python -m tf2onnx.convert --checkpoint ./ckpt/checkpoint.ckpt-210000.meta \
    --output 3DFeatNet.onnx --inputs input:0 --outputs output:0`
    # in/output nodes are unknown fails, as the ckpt files
    # do not capture the custom ops required to successfully convert to ONNX.
    ```
    - Trying to convert using PB using the --graphdef flag also does not work, as the input and output nodes of the computation graph are unknown.

Proposed actions tomorrow:
1) Figure out what the input and output nodes are
2) Learn how to import custom ops into ONNX
    
# Log Wed 28/07/21
- Attempt to learn about custom ops in TF2
- Attempt to learn about registering custom ops in ONNX runtime
- To be able to register the DH3D, Grouping and Sampling ops into ONNX, which would then be able to convert something into TensorRT.
- Alternatively, converting directly from TensorFlow to TensorRT using Saved Models may be more realistic?

- Learn how Saved Models work (especially in TensorFlow 2)

- Tried to run covnersion directly to TensorRT from .pb file but complains that TensorRT is not the same version.
- Installed TensoRT 7.xx for CUDA 11.1, also installed CUDA 11.1.

First uninstall Nvidia and Cuda completely before installing TensorRT.
```bash
sudo apt-get purge nvidia*
sudo apt-get purge cuda*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

# Then install CUDA again
sudo apt install cuda=11.1

# Then run (For Ubuntu 18.04), after downloading the package and navigating to the relevant folder
os="ubuntu1804"
tag="cuda11.1-trt7.2.3.4-ga-20210226"
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/7fa2af80.pub

sudo apt-get update
sudo aptitude install tensorrt
# Initially this will fail, so choose the option that installs the older versions of the packages.
```

This installation of TensorRT 7.x makes the program run a bit further than before, but now it runs into some errors...
Running
`(tf2onnx) shibin1995@engineer:~/ty/tutorial/TensorRT/quickstart/threeDfeatNet$ python3 tf2trt_converter.py --format pb2trt --pb_path ./export_dir/0/saved_model.pb` gives

`2021-07-28 16:15:23.490254: W tensorflow/core/grappler/utils/graph_view.cc:836] No registered 'Const' OpKernel for GPU devices compatible with node {{node save_1/filename/input}}
	 (OpKernel was found, but attributes didn't match) Requested Attributes: dtype=DT_STRING, value=Tensor<type: string shape: [] values: model>, _device="/device:GPU:1"
	.  Registered:  device='DEFAULT'; dtype in [DT_VARIANT]
`
This is possibly due to the fact that the train network is somehow being used.

# Log Thur 29/07/21
- Obtained the authors' labelled version of the 3DFeatNet network file, saved in `~/ty/autovision_coslam/src/featnet/data/sample64/saved_model.pb`. There is also a `sample128` folder.
- With this, we can run `netron saved_model.pb` to visualise the model. (it is quite big)
- The inputs:
    - `anchor:0` for the anchor point cloud
    - `is_training:0` set to _False_ so the network knows it is in inference mode
    - `detection/query_points:0` computes the attention weights for all points, instead of the 512 points used during training.
- The outputs:
    - `detection/query_points:0` (same as the input) corresponds to the x,y,z positions corresponding to the extracted features.
    - `description/features:0` outputs feature descriptors, for each point in x,y,z.
    - `detection/pts_attention:0` gives the saliency score for each point in x,y,z.

- Added auto TF and CUDA version detection into the custom ops install script.

- Tried to pull the `trt` branch of Wei Loon's `3DFeatNet` git fork, and ran into issues with linking the symbolic library

- Try to run `tb.py` for visualisation on TensorBoard.
- ```bash
    python tb.py --format pb --pb_path "../labelled_nodes/saved_model.pb" --log_dir "./tb_logs"
    tensorboard --logdir=./tb_logs
  ```

- Now we have verifed that these nodes are labelled, we can try to convert into ONNX.
    ```bash
    cd ~/ty/tutorial/TensorRT/quickstart/threeDfeatNet/3DFeatNet
    # Or wherever you have the labelled model saved
    python -m tf2onnx.convert --saved-model "../labelled_nodes/" \
    --output "../labelled_nodes/saved_model_onnx.onnx"
    ```
    - This returns 
    > tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered 'QueryBallPoint' in binary running on engineer. Make sure the Op and Kernel are registered in the binary running in this process. Note that if you are loading a saved graph which used ops from tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done before importing the graph, as contrib ops are lazily registered when the module is first accessed.
    - This points to us needing to find and register the Custom Ops somehow before calling the script.
    - Therefore, it is necessary to load the `saved_model.pb` file and parse it before changing it into an ONNX within a Python script - see `tf2onnx_converter.py` for progress on it.
    - Difficulty: In understanding how to manipulate saved models to obtain frozen graphs and then into ONNX files.

- We can also view saved models via this CLI interface `saved_model_cli show --dir ${saved_model_path} --all`
    - Its output:
    ```    
    MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
    signature_def['serving_default']:
      The given SavedModel SignatureDef contains the following input(s):
        inputs['cloud'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, -1, 3)
            name: anchor:0
        inputs['is_training'] tensor_info:
            dtype: DT_BOOL
            shape: unknown_rank
            name: is_training:0
        inputs['xyz_subset'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, -1, 3)
            name: detection/query_points:0
      The given SavedModel SignatureDef contains the following output(s):
        outputs['attention'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, -1)
            name: detection/pts_attention:0
        outputs['features'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, -1, 32)
            name: description/features:0
        outputs['xyz_out'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, -1, 3)
            name: detection/query_points:0
      Method name is: tensorflow/serving/predict
    ```
- Also attempted to run `python3 tf2trt_converter.py --format pb2trt --pb_path ./labelled_nodes/saved_model.pb`, but unfortunately ran into issues with model converison:
- `ValueError: Input 0 of node detection/conv0/bn/bn/cond/ExponentialMovingAverage/AssignMovingAvg/Switch was passed float from detection/conv0/bn/detection/conv0/bn/moments/Squeeze/ExponentialMovingAverage:0 incompatible with expected float_ref.`
- Seems like there are a few nodes which expect references, but the inputs are actual objects instead.
- Fix: Load model into TF first and change nodes, then after that attempt conversion into TensorRT
    - This proves difficult because there is a difference between tf1 and tf2 SavedModels.
        - The SavedModel is kind of a graph that saves the required operations needed for computation.
        - Output of calling 
            ```python
            model = tf.saved_model.load(PB_PATH)
            print(model.__dict__)
            ''' {
            'restore': <function _EagerSavedModelLoader.load.<locals>.<lambda> at 0x7f691ca1bf80>, 
            '_self_setattr_tracking': True, 
            '_self_unconditional_checkpoint_dependencies': 
                [TrackableReference(name='initializer', ref=<tensorflow.python.saved_model.load_v1_in_v2._Initializer object at 0x7f691c66cb10>), 
                TrackableReference(name='asset_paths', ref=ListWrapper([])), 
                TrackableReference(name='signatures', ref=_SignatureMap({'serving_default': <ConcreteFunction pruned(cloud, xyz_subset, is_training) at 0x7F69951AF390>})), 
                TrackableReference(name='variables', ref=ListWrapper([]))], 
            '_self_unconditional_dependency_names': {'initializer': <tensorflow.python.saved_model.load_v1_in_v2._Initializer object at 0x7f691c66cb10>, 
            'asset_paths': ListWrapper([]), 
            'signatures': _SignatureMap({'serving_default': <ConcreteFunction pruned(cloud, xyz_subset, is_training) at 0x7F69951AF390>}), 
            'variables': ListWrapper([])}, 
            '_self_unconditional_deferred_dependencies': {}, 
            '_self_update_uid': -1, 
            '_self_name_based_restores': set(), 
            '_self_saveable_object_factories': {}, 
            'initializer': <tensorflow.python.saved_model.load_v1_in_v2._Initializer object at 0x7f691c66cb10>, 
            'asset_paths': ListWrapper([]), 
            'signatures': _SignatureMap({'serving_default': <ConcreteFunction pruned(cloud, xyz_subset, is_training) at 0x7F69951AF390>}), 
            'variables': ListWrapper([]), 
            'tensorflow_version': '1.6.0', 
            'tensorflow_git_version': 'v1.6.0-0-gd2e24b6039', 
            'graph': <tensorflow.python.framework.func_graph.FuncGraph object at 0x7f691d75ded0>, 
            'prune': <bound method WrappedFunction.prune of <tensorflow.python.eager.wrap_function.WrappedFunction object at 0x7f691d75de90>>, 
            'graph_debug_info': 
            }
            '''  
            ```
            which is quite interesting, showing that the main computation is in the `graph` attribute of this given object.           
    - Learning more about the content of a SavedModel in TF2 and how to make it print out its nodes.
    - If possible, try to regenerate / rebuild the SavedModel in TF2.
    
# Log Fri 30/07/21
- Try harder to convert the model into its relevant type.
- Opened the model in TensorBoard for more information.
- Look into copying graph in TensorFlow2 bit by bit?
    - TF1 [link](https://github.com/davidsandberg/facenet/issues/161) for resolving "expected `float_ref` but got `float`"
    ```python
    saver = tf.train.import_meta_graph(os.path.join(os.path.expanduser(args.model_dir), 
        'model-' + os.path.basename(os.path.normpath(args.model_dir)) + '.meta'), clear_devices=True)
    tf.get_default_session().run(tf.global_variables_initializer())
    tf.get_default_session().run(tf.local_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(os.path.expanduser(args.model_dir)))

    output_node_names = 'embeddings'

    # for fixing the bug of batch norm
    gd = sess.graph.as_graph_def()
    for node in gd.node:            
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, output_node_names.split(","))
    tf.train.write_graph(converted_graph_def, args.output_dir, args.output_filename, as_text=False)
    ```
    
- Ran `python -m tf2onnx.convert --saved-model ../labelled_nodes/ --output model.onnx --load_op_libraries ./tf_ops/grouping/tf_grouping_so.so,./tf_ops/sampling/tf_sampling_so.so`
- The difference being the linking of the `.so` libraries within the `tf2onnx.convert` command.
- However, get the same result:
    `ValueError: Input 0 of node detection/conv0/bn/bn/cond/ExponentialMovingAverage/AssignMovingAvg/Switch was passed float from detection/conv0/bn/detection/conv0/bn/moments/Squeeze/        ExponentialMovingAverage:0 incompatible with expected float_ref.`

- So even if the custom ops can be registered within `tf2onnx`, we still need to resolve the incorrect definition to the graph.

- Created a `tf1env` Conda env, using `tensorflow-gpu=1.14` and trying to thus modify the graph as such using that.
- Had to re-compile the custom ops given the different Tensorflow version.
- Import the saved model by calling `model = tf.compat.v1.saved_model.load(sess, tags=["serve"], export_dir=args.pb_path)`; with the `tags` parameter discovered by running `saved_model_cli` as done earlier..

- Moving on to obtain a modifiable MetaGraphDef for the output `model` object.
- This is attained using the `tf_graphconverter.py` script in `~quickstart/threeDfeatNet`.
- Using tensorflow 1, imported and set a graph, and used the `saved_model.builder.SavedModelBuilder()` class to save not just the modified nodes, but also the variables.
- This is able to be reflected using `saved_model_cli`.
- The next step would be to figure out how to convert it into an ONNX or TensorRT model.
- However, this raises another error:
```
Cannot assign a device for operation detection/QueryBallPoint: Could not satisfy explicit device specification '/device:CPU:0' because no supported kernel for CPU devices is available.
Colocation Debug Info:
Colocation group had the following types and supported devices: 
Root Member(assigned_device_name_index_=-1 requested_device_name_='/device:CPU:0' assigned_device_name_='' resource_device_name_='' supported_device_types_=[GPU] possible_devices_=[]
QueryBallPoint: GPU 

Colocation members, user-requested devices, and framework assigned devices, if any:
  detection/QueryBallPoint (QueryBallPoint) /device:CPU:0
360K	labelled_nodes/saved_model.pb

Op: QueryBallPoint
Node attrs: radius=2, nsample=64
Registered kernels:
  device='GPU'

[[node detection/QueryBallPoint (defined at /site-packages/tf2onnx/tf_loader.py:368) ]]
```

# Log Mon 02/08/21
- Attempt to convert the modified SavedModel using `tf2onnx.convert` in a TensorFlow2 Conda environment, but still get `ValueError: Input 0 of node detection/conv0/bn/bn/cond/ExponentialMovingAverage/AssignMovingAvg/Switch was passed float from detection/conv0/bn/detection/conv0/bn/moments/Squeeze/ExponentialMovingAverage:0 incompatible with expected float_ref`, which is the same as before.
- Going to attempt to resolve this by changing saving the original TF1 SavedModel into TF2.
- This did not work.

- Attempt: to try to re-train 3DFeatNet in TensorFlow 2.
- Steps we can take to "modernize" 3DFeatNet
    1) Add names to the input and output nodes of the open-source 3DFeatNet code. 
    2) Refactor training script to TF2
    3) Train
    4) Evaluate Results

## Progress Update
We tried using the Saved Model with labelled nodes as Ben provided. However, we ran into multiple further issues with this file.
The main problem was (ad verbatim) ```ValueError: Input 0 of node detection/conv0/bn/bn/cond/ExponentialMovingAverage/AssignMovingAvg/Switch was passed float from detection/conv0/bn/detection/conv0/bn/moments/Squeeze/ExponentialMovingAverage:0 incompatible with expected float_ref```. This might be due to the nature of the nodes used in the computation graph, so we attempted to copy the graph in TF1 to replace the problematic nodes. 

However, tf2onnx conversion did not recognise the custom ops registered for GPU, and attempted to run the custom ops using CPU (which obviously didn't work). On the other hand, trying to refactor the provided TF1 Saved Model in TF2 seemed not possible, as saved models are different in all but name in TF1 vs TF2.

We collected the fully trained DH3D model today and will work on testing that. In addition, we will also try to refactor the 3DFeatNet training script into TF2 and retrain the model, as it seems to be the only sustainable way forward for integration with modern tools. To that end, is there a full 3DFeatNet training script including the labels?


# Log Fri 06/08/21
- Converted the Tensorflow 1.x 3DFeatNet repo to be able to run in TF2.5, under the `tensorflow2` branch of the Git repo. However, it is still largely running off `tf.compat.v1` idioms and not with TF2 syntax. Therefore, next week the idea is to be able to convert the code to be in TF2.

- Figure out how to map `tf_slim` operations into `tf.keras` layers, and to convert the TF1 Graph into something more of a `Model` object.












