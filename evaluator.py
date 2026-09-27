import numpy as np

def _hpwl_impl(pins):
    n = pins.shape[0]
    if n == 0:
        return 0.0
    min_x = pins[0, 0]
    max_x = pins[0, 0]
    min_y = pins[0, 1]
    max_y = pins[0, 1]
    for i in range(1, n):
        x = pins[i, 0]
        y = pins[i, 1]
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y
    return (max_x - min_x) + (max_y - min_y)

try:
    from numba import njit

    _numba_hpwl = njit(cache=True)(_hpwl_impl)
except (ImportError, RuntimeError):
    _numba_hpwl = _hpwl_impl

class Evaluator:
    def __init__(self, placedb, grid_size=64, fine_grids_per_coarse_cell=None):
        self.placedb = placedb
        self.node_info = placedb.node_info
        self.net_info = placedb.net_info
        self.node_id_to_name = placedb.node_id_to_name
        self.chip_height = placedb.chip_height
        self.chip_width = placedb.chip_width
        self.num_nodes = len(self.node_id_to_name)
        self.grid_size = grid_size
        self.cell_w = self.chip_width / grid_size
        self.cell_h = self.chip_height / grid_size
        self.fine_grids_per_coarse_cell = fine_grids_per_coarse_cell

        self.node_name_to_idx = {name: i for i, name in enumerate(self.node_id_to_name)}

        self.net_to_nodes = []
        self.net_weights = []
        self.net_pin_offsets = []
        for net_name in self.net_info:
            net = self.net_info[net_name]
            nodes = list(net["nodes"].keys())
            self.net_to_nodes.append(nodes)
            self.net_weights.append(net.get("weight", 1.0))
            offsets = []
            for node_name in nodes:
                ni = self.node_info[node_name]
                nt = net["nodes"][node_name]
                offsets.append((
                    ni["x"] / 2.0 + nt["x_offset"],
                    ni["y"] / 2.0 + nt["y_offset"]
                ))
            self.net_pin_offsets.append(offsets)

        self.node_to_nets = [[] for _ in range(self.num_nodes)]
        for net_idx, nodes in enumerate(self.net_to_nodes):
            for node_name in nodes:
                node_idx = self.node_name_to_idx[node_name]
                self.node_to_nets[node_idx].append(net_idx)

    def _get_position(self, solution, idx):
        xc, yc, xf, yf = solution[idx]
        if self.fine_grids_per_coarse_cell is not None:
            gx = xc * self.fine_grids_per_coarse_cell + xf
            gy = yc * self.fine_grids_per_coarse_cell + yf
            return gx * self.cell_w, gy * self.cell_h
        return xc * self.cell_w, yc * self.cell_h

    def get_positions(self, solution):
        positions = {}
        for i, node_name in enumerate(self.node_id_to_name):
            px, py = self._get_position(solution, i)
            positions[node_name] = (px, py)
        return positions

    def compute_hpwl(self, solution):
        hpwl_total = 0.0
        for net_idx, nodes in enumerate(self.net_to_nodes):
            pins = []
            for node_i, node_name in enumerate(nodes):
                node_idx = self.node_name_to_idx[node_name]
                px, py = self._get_position(solution, node_idx)
                ox, oy = self.net_pin_offsets[net_idx][node_i]
                pins.append((px + ox, py + oy))
            if len(pins) <= 1:
                continue
            pins_arr = np.array(pins, dtype=np.float64)
            hpwl = _numba_hpwl(pins_arr)
            hpwl_total += hpwl * self.net_weights[net_idx]
        return hpwl_total

    def compute_overlap(self, solution):
        overlaps = 0.0
        n = self.num_nodes
        for i in range(n):
            name_i = self.node_id_to_name[i]
            ni = self.node_info[name_i]
            px, py = self._get_position(solution, i)
            wi = ni["x"]
            hi = ni["y"]
            for j in range(i + 1, n):
                name_j = self.node_id_to_name[j]
                nj = self.node_info[name_j]
                pxj, pyj = self._get_position(solution, j)
                wj = nj["x"]
                hj = nj["y"]
                inter_w = min(px + wi, pxj + wj) - max(px, pxj)
                inter_h = min(py + hi, pyj + hj) - max(py, pyj)
                if inter_w > 0 and inter_h > 0:
                    overlaps += inter_w * inter_h
        return overlaps

    def compute_rudy_congestion(self, solution, top_ratio=0.1):
        grid_cong = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        for net_idx, nodes in enumerate(self.net_to_nodes):
            if len(nodes) <= 1:
                continue
            pins = []
            for node_i, node_name in enumerate(nodes):
                node_idx = self.node_name_to_idx[node_name]
                px, py = self._get_position(solution, node_idx)
                ox, oy = self.net_pin_offsets[net_idx][node_i]
                pins.append((px + ox, py + oy))
            pins_arr = np.array(pins, dtype=np.float64)
            min_x = pins_arr[:, 0].min()
            max_x = pins_arr[:, 0].max()
            min_y = pins_arr[:, 1].min()
            max_y = pins_arr[:, 1].max()
            w = max_x - min_x
            h = max_y - min_y
            if w == 0 or h == 0:
                continue
            impact = (w + h) / (w * h)
            g_min_x = max(0, int(min_x / self.cell_w))
            g_max_x = min(self.grid_size - 1, int(max_x / self.cell_w))
            g_min_y = max(0, int(min_y / self.cell_h))
            g_max_y = min(self.grid_size - 1, int(max_y / self.cell_h))
            for gx in range(g_min_x, g_max_x + 1):
                for gy in range(g_min_y, g_max_y + 1):
                    grid_cong[gx, gy] += impact
        flat = grid_cong.flatten()
        k = max(1, int(len(flat) * top_ratio))
        top_k = np.partition(flat, -k)[-k:]
        return float(top_k.mean())

    def compute_fitness(self, solution, lambda_cong=1e-4, lambda_overlap=1e6):
        hpwl = self.compute_hpwl(solution)
        overlap = self.compute_overlap(solution)
        congestion = self.compute_rudy_congestion(solution)
        cost = hpwl + lambda_cong * congestion + lambda_overlap * overlap
        return -cost, hpwl, overlap, congestion

    def compute_rudy_congestion_from_phenotype(self, phenotype, top_ratio=0.1):

        cong_grid_size = 64
        grid_cong = np.zeros((cong_grid_size, cong_grid_size), dtype=np.float64)
        cell_w = self.chip_width / cong_grid_size
        cell_h = self.chip_height / cong_grid_size

        for net_idx, nodes in enumerate(self.net_to_nodes):
            if len(nodes) <= 1:
                continue
            pins = []
            for node_i, node_name in enumerate(nodes):
                node_idx = self.node_name_to_idx[node_name]
                gx, gy = phenotype[node_idx]
                px = gx * self.cell_w
                py = gy * self.cell_h
                ox, oy = self.net_pin_offsets[net_idx][node_i]
                pins.append((px + ox, py + oy))
            pins_arr = np.array(pins, dtype=np.float64)
            min_x = pins_arr[:, 0].min()
            max_x = pins_arr[:, 0].max()
            min_y = pins_arr[:, 1].min()
            max_y = pins_arr[:, 1].max()
            w = max_x - min_x
            h = max_y - min_y
            if w == 0 or h == 0:
                continue
            impact = (w + h) / (w * h)
            g_min_x = max(0, int(min_x / cell_w))
            g_max_x = min(cong_grid_size - 1, int(max_x / cell_w))
            g_min_y = max(0, int(min_y / cell_h))
            g_max_y = min(cong_grid_size - 1, int(max_y / cell_h))
            for gx in range(g_min_x, g_max_x + 1):
                for gy in range(g_min_y, g_max_y + 1):
                    grid_cong[gx, gy] += impact
        flat = grid_cong.flatten()
        k = max(1, int(len(flat) * top_ratio))
        top_k = np.partition(flat, -k)[-k:]
        return float(top_k.mean())

    def compute_wire_mask_for_node(self, solution, node_idx):
        node_name = self.node_id_to_name[node_idx]
        ni = self.node_info[node_name]
        w = ni["x"]
        h = ni["y"]
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        for net_idx in self.node_to_nets[node_idx]:
            nodes = self.net_to_nodes[net_idx]
            if len(nodes) <= 1:
                continue
            pins = []
            node_i_in_net = None
            for node_i, nname in enumerate(nodes):
                nidx = self.node_name_to_idx[nname]
                if nidx == node_idx:
                    node_i_in_net = node_i
                px, py = self._get_position(solution, nidx)
                ox, oy = self.net_pin_offsets[net_idx][node_i]
                pins.append((px + ox, py + oy))
            if node_i_in_net is None:
                continue
            ox, oy = self.net_pin_offsets[net_idx][node_i_in_net]
            for gx in range(self.grid_size):
                for gy in range(self.grid_size):
                    px = gx * self.cell_w
                    py = gy * self.cell_h
                    pins[node_i_in_net] = (px + ox, py + oy)
                    pins_arr = np.array(pins, dtype=np.float64)
                    hpwl = _numba_hpwl(pins_arr)
                    mask[gx, gy] += hpwl * self.net_weights[net_idx]
        return mask
