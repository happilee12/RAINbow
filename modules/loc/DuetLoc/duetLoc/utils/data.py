import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import random
import torch

class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size, use_gpu=False, preload_all=False):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}
        self.use_gpu = use_gpu
        self.preload_all = preload_all
        
        if self.preload_all:
            print(f"Preloading all features from {img_ft_file} to {'GPU' if use_gpu else 'CPU'} memory...")
            with h5py.File(self.img_ft_file, 'r') as f:
                for key in f.keys():
                    ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                    if self.use_gpu:
                        self._feature_store[key] = torch.from_numpy(ft).cuda()
                    else:
                        self._feature_store[key] = ft
            print(f"Preloaded {len(self._feature_store)} features")
        

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                if self.use_gpu:
                    ft = torch.from_numpy(ft).cuda()
                self._feature_store[key] = ft
        return ft
        
    def preload_features(self, scan_viewpoint_list):
        """
        Preload a specific set of features to GPU or CPU memory
        
        Args:
            scan_viewpoint_list: List of (scan, viewpoint) tuples to preload
        """
        if self.preload_all:
            print("All features already preloaded, skipping partial preload")
            return
            
        with h5py.File(self.img_ft_file, 'r') as f:
            for scan, viewpoint in scan_viewpoint_list:
                key = '%s_%s' % (scan, viewpoint)
                if key not in self._feature_store and key in f:
                    ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                    if self.use_gpu:
                        self._feature_store[key] = torch.from_numpy(ft).cuda()
                    else:
                        self._feature_store[key] = ft
        
        print(f"Preloaded {len(self._feature_store)} features to {'GPU' if self.use_gpu else 'CPU'} memory")
        
    def clear_feature_cache(self):
        """Clear the feature cache to free memory"""
        self._feature_store = {}

class ImageFeaturesDB2(object):
    def __init__(self, img_ft_files, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_files
        self._feature_stores = {}
        for name in img_ft_files:
            self._feature_stores[name] = {}
            with h5py.File(name, 'r') as f:
                for key in f.keys():
                    ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                    self._feature_stores[name][key] = ft 
        self.env_names = list(self._feature_stores.keys())
        print(self.env_names)
        

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        env_name = random.choice(self.env_names)
        if key in self._feature_stores[env_name]:
            ft = self._feature_stores[env_name][key]
        else:
            with h5py.File(env_name, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_stores[env_name][key] = ft
        return ft

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

