from models import feat3dnet, feat3dnet_tf2

networks_map = {'3DFeatNet': feat3dnet.Feat3dNet,
                '3DFeatNet_tf2': feat3dnet_tf2.Feat3dNet
                }


def get_network(name):

    model = networks_map[name]
    return model
