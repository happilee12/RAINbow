from collections import defaultdict
import numpy as np
import torch

MAX_DIST = 30
MAX_STEP = 10

def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist

def calc_position_distances_gpu(positions_a, positions_b):
    # print("calc_position_distances_gpu")
    # print(positions_a.shape, positions_b.shape)
    # print(positions_a, positions_b)
    positions_a = torch.from_numpy(positions_a).float()
    positions_b = torch.from_numpy(positions_b).float()
    diff = positions_b - positions_a
    distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
    return distances.numpy()

def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz/xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist

def calculate_vp_rel_pos_fts_gpu(cur_pos, target_positions, base_heading=0, base_elevation=0):
    # print("calculate_vp_rel_pos_fts_gpu")
    dx = target_positions[:, 0] - cur_pos[0]
    dy = target_positions[:, 1] - cur_pos[1] 
    dz = target_positions[:, 2] - cur_pos[2]
    
    # Compute distances
    xy_dist = torch.clamp(torch.sqrt(dx**2 + dy**2), min=1e-8)
    xyz_dist = torch.clamp(torch.sqrt(dx**2 + dy**2 + dz**2), min=1e-8)
    
    # Compute heading angles
    heading = torch.asin(dx/xy_dist)  # [-pi/2, pi/2]
    # Handle the case where b[1] < a[1] (y-axis condition)
    heading = torch.where(dy < 0, torch.pi - heading, heading)
    heading -= base_heading
    
    # Compute elevation angles
    elevation = torch.asin(dz/xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation
    
    return heading, elevation, xyz_dist

def get_angle_fts_gpu(headings, elevations, angle_feat_size):
    # print("get_angle_fts_gpu")
    ang_fts = [torch.sin(headings), torch.cos(headings), torch.sin(elevations), torch.cos(elevations)]
    ang_fts = torch.stack(ang_fts, dim=1).float()
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = torch.cat([ang_fts] * num_repeats, dim=1)
    return ang_fts

def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts


class FloydGraph(object):
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)
    
    def distances_batch(self, x, targets):
        """
        Get distances from x to multiple targets efficiently
        
        Args:
            x: source viewpoint
            targets: list of target viewpoints
            
        Returns:
            distances: numpy array of distances
        """
        distances = np.zeros(len(targets), dtype=np.float32)
        for i, y in enumerate(targets):
            if x == y:
                distances[i] = 0
            else:
                distances[i] = self._dis[x][y]
        return distances
    
    def path_lengths_batch(self, x, targets):
        """
        Get path lengths from x to multiple targets efficiently
        
        Args:
            x: source viewpoint
            targets: list of target viewpoints
            
        Returns:
            path_lengths: numpy array of path lengths
        """
        path_lengths = np.zeros(len(targets), dtype=np.float32)
        for i, y in enumerate(targets):
            if x == y:
                path_lengths[i] = 0
            else:
                path_lengths[i] = len(self.path(x, y))
        return path_lengths


class GraphMap(object):
    def __init__(self, start_vp):
        self.start_vp = start_vp    # start viewpoint

        self.node_positions = {}             # viewpoint to position (x, y, z)
        self.graph = FloydGraph()   # shortest path graph
        self.node_embeds = {}       # {viewpoint: feature (sum feature, count)}
        self.node_stop_scores = {}  # {viewpoint: prob}
        self.node_nav_scores = {}   # {viewpoint: {t: prob}}
        self.node_step_ids = {}
        self.history_embeds = {}

    # def update_graph_naive(self, ob):
    #     self.node_positions[ob['viewpoint']] = ob['position']
    #     for cc in ob['candidate']:
    #         self.node_positions[cc['viewpointId']] = cc['position']
    #         dist = calc_position_distances_gpu(ob['position'], cc['position'])
    #         self.graph.add_edge(ob['viewpoint'], cc['viewpointId'], dist)
    #     self.graph.update(ob['viewpoint'])

    def update_graph(self, ob):
        # Vectorized version of update_graph
        self.node_positions[ob['viewpoint']] = ob['position']
        
        if not ob['candidate']:
            self.graph.update(ob['viewpoint'])
            return
        
        # Extract candidate positions and IDs
        candidate_positions = np.array([cc['position'] for cc in ob['candidate']])
        candidate_ids = [cc['viewpointId'] for cc in ob['candidate']]
        
        # Vectorized distance calculation
        current_pos = np.array(ob['position'])
        current_pos_expanded = np.tile(current_pos, (len(candidate_positions), 1))
        distances = calc_position_distances_gpu(current_pos_expanded, candidate_positions)
        
        # Batch update node positions
        for cand_id, pos in zip(candidate_ids, ob['candidate']):
            self.node_positions[cand_id] = pos['position']
        
        # Batch add edges
        for cand_id, dist in zip(candidate_ids, distances):
            self.graph.add_edge(ob['viewpoint'], cand_id, dist)
        
        self.graph.update(ob['viewpoint'])

    def update_node_embed(self, vp, embed, rewrite=False):
        if rewrite:
            self.node_embeds[vp] = [embed, 1]
        else:
            if vp in self.node_embeds:
                self.node_embeds[vp][0] = self.node_embeds[vp][0] + embed
                self.node_embeds[vp][1] = self.node_embeds[vp][1] + 1
            else:
                self.node_embeds[vp] = [embed, 1]
    
    def update_history_embed(self, embeds, vpids):
        for vpid, embed in zip(vpids, embeds):
            self.history_embeds[vpid] = embed

    def get_history_embed(self, vp):
        if vp not in self.history_embeds:
            return self.get_node_embed(vp)
        return self.history_embeds[vp]

    def get_node_embed(self, vp):
        return self.node_embeds[vp][0] / self.node_embeds[vp][1]

    def get_pos_fts_naive(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.node_positions[cur_vp], self.node_positions[vp],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.graph.distance(cur_vp, vp) / MAX_DIST, \
                    len(self.graph.path(cur_vp, vp)) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def get_pos_fts(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):
        # Create mask for valid viewpoints (not None)
        valid_mask = np.array([vp is not None for vp in gmap_vpids], dtype=bool)
        
        # Initialize arrays with zeros
        n_vps = len(gmap_vpids)
        rel_angles = np.zeros((n_vps, 2), dtype=np.float32)
        rel_dists = np.zeros((n_vps, 3), dtype=np.float32)
        
        if np.any(valid_mask):
            # Get valid viewpoint IDs and their positions
            valid_vpids = [vp for vp in gmap_vpids if vp is not None]
            valid_positions = np.array([self.node_positions[vp] for vp in valid_vpids])
            
            # Vectorized calculation for valid viewpoints
            rel_headings, rel_elevations, rel_distances = calculate_vp_rel_pos_fts(
                self.node_positions[cur_vp], valid_positions,
                base_heading=cur_heading, base_elevation=cur_elevation
            )
            
            # Fill in the results for valid viewpoints
            rel_angles[valid_mask, 0] = rel_headings
            rel_angles[valid_mask, 1] = rel_elevations
            
            # Vectorized distance calculations
            rel_dists[valid_mask, 0] = rel_distances / MAX_DIST
            
            # Batch calculate graph distances and path lengths
            graph_distances = self.graph.distances_batch(cur_vp, valid_vpids)
            path_lengths = self.graph.path_lengths_batch(cur_vp, valid_vpids)
            
            # Fill in the results for valid viewpoints
            rel_dists[valid_mask, 1] = graph_distances / MAX_DIST
            rel_dists[valid_mask, 2] = path_lengths / MAX_STEP
        
        # Generate angle features using the existing function
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def get_pos_fts_gpu(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):
        # print("get_pos_fts_gpu")
        # Create mask for valid viewpoints (not None)
        valid_mask = torch.tensor([vp is not None for vp in gmap_vpids], dtype=torch.bool, device='cuda')
        
        # Initialize tensors with zeros
        n_vps = len(gmap_vpids)
        rel_angles = torch.zeros((n_vps, 2), dtype=torch.float32, device='cuda')
        rel_dists = torch.zeros((n_vps, 3), dtype=torch.float32, device='cuda')
        
        if torch.any(valid_mask):
            # Get valid viewpoint IDs and their positions
            valid_vpids = [vp for vp in gmap_vpids if vp is not None]
            valid_positions = torch.tensor([self.node_positions[vp] for vp in valid_vpids], 
                                         dtype=torch.float32, device='cuda')
            
            # Convert current position to GPU tensor
            cur_pos = torch.tensor(self.node_positions[cur_vp], dtype=torch.float32, device='cuda')
            
            # Vectorized calculation for valid viewpoints on GPU
            rel_headings, rel_elevations, rel_distances = calculate_vp_rel_pos_fts_gpu(
                cur_pos, valid_positions,
                base_heading=cur_heading, base_elevation=cur_elevation
            )
            
            # Fill in the results for valid viewpoints
            valid_indices = torch.where(valid_mask)[0]
            rel_angles[valid_indices, 0] = rel_headings
            rel_angles[valid_indices, 1] = rel_elevations
            
            # Vectorized distance calculations
            rel_dists[valid_indices, 0] = rel_distances / MAX_DIST
            
            # Batch calculate graph distances and path lengths (still CPU for now)
            graph_distances = self.graph.distances_batch(cur_vp, valid_vpids)
            path_lengths = self.graph.path_lengths_batch(cur_vp, valid_vpids)
            
            # Convert to GPU tensors
            graph_distances = torch.tensor(graph_distances, dtype=torch.float32, device='cuda')
            path_lengths = torch.tensor(path_lengths, dtype=torch.float32, device='cuda')
            
            # Fill in the results for valid viewpoints
            rel_dists[valid_indices, 1] = graph_distances / MAX_DIST
            rel_dists[valid_indices, 2] = path_lengths / MAX_STEP
        
        # Generate angle features using GPU function
        rel_ang_fts = get_angle_fts_gpu(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return torch.cat([rel_ang_fts, rel_dists], dim=1)

    def save_to_json(self):
        nodes = {}
        for vp, pos in self.node_positions.items():
            nodes[vp] = {
                'location': pos,    # (x, y, z)
                'visited': self.graph.visited(vp),
            }
            if nodes[vp]['visited']:
                nodes[vp]['stop_prob'] = self.node_stop_scores[vp]['stop']
                nodes[vp]['og_objid'] = self.node_stop_scores[vp]['og']
            else:
                nodes[vp]['nav_prob'] = self.node_nav_scores[vp]

        edges = []
        for k, v in self.graph._dis.items():
            for kk in v.keys():
                edges.append((k, kk))
                
        return {'nodes': nodes, 'edges': edges}
    
    
    