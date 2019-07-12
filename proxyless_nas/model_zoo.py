import copy
import os
import json

from .nas_modules import ProxylessNASNets


def proxyless_net(id):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"proxyless_net{id}.config")
    net_config_json = json.load(open(config_path, 'r'))
    net = ProxylessNASNets.build_from_config(net_config_json)
    net.set_bn_param(
        bn_momentum=net_config_json['bn']['momentum'],
        bn_eps=net_config_json['bn']['eps'])
    return net

