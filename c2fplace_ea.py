import copy
import os
import random

import numpy as np

from place_db import PlaceDB
from evaluator import Evaluator, _numba_hpwl

class Individual:
    def __init__(self, genotype, phenotype=None, fitness=None, hpwl=None, overlap=None, congestion=None):
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness
        self.hpwl = hpwl
        self.overlap = overlap
        self.congestion = congestion

class ValidationResult:
    def __init__(self, is_valid, errors, overlap):
        self.is_valid = is_valid
        self.errors = errors
        self.overlap = overlap

class C2FPlaceEA:

    def __init__(self, placedb, pop_size=20,
                 tournament_size=5, mutation_rate=0.3,
                 coarse_grid_size=224,
                 lambda_cong=1e-4, lambda_overlap=1e6,
                 greedy_iter=2, stage1_iters=100, stage2_iters=100, stage3_iters=0,
                 stage3_order="node_id", stage3_mode="both_2",
                 ripup_ratio=0.2,
                 ripup_strategy="random",
                 bbox_order="count_desc",
                 replace_order="auto",
                 random_init_order=False,
                 ripup_ratio_strategy="fixed",
                 seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.placedb = placedb

        self.global_grid_size = placedb.global_grid_size
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate

        if coarse_grid_size > self.global_grid_size:
            print(f"Warning: coarse_grid_size ({coarse_grid_size}) > global_grid_size ({self.global_grid_size}), adjusting to {self.global_grid_size}")
            coarse_grid_size = self.global_grid_size
        self.coarse_grid_size = coarse_grid_size

        self.fine_grids_per_coarse_cell = self.global_grid_size // coarse_grid_size
        self.lambda_cong = lambda_cong
        self.lambda_overlap = lambda_overlap
        self.greedy_iter = greedy_iter
        self.stage1_iters = stage1_iters
        self.stage2_iters = stage2_iters
        self.stage3_iters = stage3_iters
        self.stage3_order = stage3_order
        self.stage3_mode = stage3_mode
        # if ripup_strategy == "bbox" and placedb.benchmark in {"adaptec3", "bigblue3", "bigblue4"}:
        #     ripup_ratio = 0.05
        self.ripup_ratio = ripup_ratio
        valid_ripup_strategies = {"random", "bbox"}
        valid_bbox_orders = {"random", "count_desc", "count_asc", "area_asc", "node_id"}
        valid_replace_orders = {"auto", "preserve", "node_id", "random", "area_group"}
        valid_ripup_ratio_strategies = {"fixed", "random_stage", "decile_random"}
        valid_stage3_orders = {"hpwl", "random", "node_id"}
        valid_stage3_modes = {"coarse", "fine", "both_1", "both_2"}
        if ripup_strategy not in valid_ripup_strategies:
            raise ValueError(f"ripup_strategy must be one of {sorted(valid_ripup_strategies)}")
        if bbox_order not in valid_bbox_orders:
            raise ValueError(f"bbox_order must be one of {sorted(valid_bbox_orders)}")
        if replace_order not in valid_replace_orders:
            raise ValueError(f"replace_order must be one of {sorted(valid_replace_orders)}")
        if ripup_ratio_strategy not in valid_ripup_ratio_strategies:
            raise ValueError(f"ripup_ratio_strategy must be one of {sorted(valid_ripup_ratio_strategies)}")
        if stage3_order not in valid_stage3_orders:
            raise ValueError(f"stage3_order must be one of {sorted(valid_stage3_orders)}")
        if stage3_mode not in valid_stage3_modes:
            raise ValueError(f"stage3_mode must be one of {sorted(valid_stage3_modes)}")
        self.ripup_strategy = ripup_strategy
        self.bbox_order = bbox_order
        self.replace_order = replace_order
        self.random_init_order = random_init_order
        self.ripup_ratio_strategy = ripup_ratio_strategy

        self.evaluator = Evaluator(
            placedb,
            grid_size=self.global_grid_size,
            fine_grids_per_coarse_cell=self.fine_grids_per_coarse_cell,
        )
        self.num_nodes = self.evaluator.num_nodes
        self.node_id_to_name = self.evaluator.node_id_to_name
        self.node_info = self.evaluator.node_info
        self.chip_width = placedb.chip_width
        self.chip_height = placedb.chip_height

        self.node_name_to_idx = self.evaluator.node_name_to_idx
        self.node_to_nets = self.evaluator.node_to_nets

        self.global_cell_width = self.chip_width / self.global_grid_size
        self.global_cell_height = self.chip_height / self.global_grid_size
        self.coarse_cell_width = self.chip_width / coarse_grid_size
        self.coarse_cell_height = self.chip_height / coarse_grid_size

        print(f"Global grid: {self.global_grid_size}x{self.global_grid_size}, Coarse grid: {coarse_grid_size}x{coarse_grid_size}, Fine grids per coarse cell: {self.fine_grids_per_coarse_cell}x{self.fine_grids_per_coarse_cell}")

        self._compute_connected_areas()

        self.node_areas = []
        for i in range(self.num_nodes):
            ni = self.node_info[self.node_id_to_name[i]]
            self.node_areas.append(ni["x"] * ni["y"])
        self.total_area = sum(self.node_areas)
        print(f"Total macro area: {self.total_area:.2e}, Canvas area: {self.chip_width * self.chip_height:.2e}, Ratio: {self.total_area/(self.chip_width*self.chip_height):.3f}")
        self._precompute_node_grid_spans()
        self._verbose_mutation = False

    def _precompute_node_grid_spans(self):
        self.node_grid_spans = []
        for idx in range(self.num_nodes):
            node_name = self.node_id_to_name[idx]
            ni = self.node_info[node_name]
            w_g = int(np.ceil(ni["x"] / self.global_cell_width))
            h_g = int(np.ceil(ni["y"] / self.global_cell_height))
            w_c = max(1, int(np.ceil(w_g / self.fine_grids_per_coarse_cell)))
            h_c = max(1, int(np.ceil(h_g / self.fine_grids_per_coarse_cell)))
            max_xf = min(self.fine_grids_per_coarse_cell - 1, w_c * self.fine_grids_per_coarse_cell - w_g)
            max_yf = min(self.fine_grids_per_coarse_cell - 1, h_c * self.fine_grids_per_coarse_cell - h_g)
            if max_xf < 0 or max_yf < 0:
                raise ValueError(f"{node_name} cannot fit in its coarse grid footprint")
            self.node_grid_spans.append({
                "w_g": w_g,
                "h_g": h_g,
                "w_c": w_c,
                "h_c": h_c,
                "max_xf": max_xf,
                "max_yf": max_yf,
            })

    def validate_individual(self, individual, overlap_tolerance=0.0):
        errors = []
        phenotype = individual.phenotype
        if phenotype is None:
            errors.append("missing phenotype")
            return ValidationResult(False, errors, None)
        if len(phenotype) != self.num_nodes:
            errors.append(f"phenotype length {len(phenotype)} does not match node count {self.num_nodes}")

        placed_modules = []
        overlap = 0.0
        for idx in range(min(len(phenotype), self.num_nodes)):
            pos = phenotype[idx]
            node_name = self.node_id_to_name[idx]
            if pos is None:
                errors.append(f"{node_name} has no position")
                continue
            gx, gy = pos
            if gx is None or gy is None:
                errors.append(f"{node_name} has an incomplete position")
                continue
            x = gx * self.global_cell_width
            y = gy * self.global_cell_height
            width = self.node_info[node_name]["x"]
            height = self.node_info[node_name]["y"]
            if x < 0 or y < 0 or x + width > self.chip_width or y + height > self.chip_height:
                errors.append(
                    f"{node_name} is out of bounds: ({x:.2f}, {y:.2f}, {width:.2f}, {height:.2f})"
                )
            for _, px, py, pw, ph in placed_modules:
                inter_w = min(x + width, px + pw) - max(x, px)
                inter_h = min(y + height, py + ph) - max(y, py)
                if inter_w > 0 and inter_h > 0:
                    overlap += inter_w * inter_h
            placed_modules.append((node_name, x, y, width, height))

        if overlap > overlap_tolerance:
            errors.append(f"placement overlap {overlap:.6g} exceeds tolerance {overlap_tolerance:.6g}")
        return ValidationResult(len(errors) == 0, errors, overlap)

    def _compute_connected_areas(self):

        self.connected_areas = []
        for i in range(self.num_nodes):
            area = 0.0
            for net_idx in self.node_to_nets[i]:
                nodes = self.evaluator.net_to_nodes[net_idx]
                for node_name in nodes:
                    nidx = self.node_name_to_idx[node_name]
                    ni = self.node_info[node_name]
                    area += ni["x"] * ni["y"]
            self.connected_areas.append(area)

    def _bbox_critical_counts(self, phenotype):
        key_counts = {}
        for net_idx, nodes in enumerate(self.evaluator.net_to_nodes):
            if len(nodes) <= 1:
                continue

            pins = []
            node_indices = []
            for node_i, node_name in enumerate(nodes):
                node_idx = self.node_name_to_idx[node_name]
                gx, gy = phenotype[node_idx]
                px = gx * self.global_cell_width
                py = gy * self.global_cell_height
                ox, oy = self.evaluator.net_pin_offsets[net_idx][node_i]
                pins.append((px + ox, py + oy))
                node_indices.append(node_idx)

            if len(pins) <= 1:
                continue

            pins_arr = np.array(pins, dtype=np.float64)
            x_vals = pins_arr[:, 0]
            y_vals = pins_arr[:, 1]
            endpoint_masks = (
                np.isclose(x_vals, x_vals.min()),
                np.isclose(x_vals, x_vals.max()),
                np.isclose(y_vals, y_vals.min()),
                np.isclose(y_vals, y_vals.max()),
            )

            extrema_nodes = set()
            for mask in endpoint_masks:
                for pin_idx in np.where(mask)[0]:
                    extrema_nodes.add(node_indices[int(pin_idx)])
            for idx in extrema_nodes:
                key_counts[idx] = key_counts.get(idx, 0) + 1
        return key_counts

    def _select_bbox_ripup_indices(self, genotype):

        phenotype = self._genotype_to_phenotype(genotype)
        key_counts = self._bbox_critical_counts(phenotype)

        if not key_counts:
            self._last_candidate_count = 1
            return [random.randrange(self.num_nodes)]

        ripup_indices = sorted(key_counts.keys())
        if self.bbox_order == "random":
            random.shuffle(ripup_indices)
        elif self.bbox_order == "count_desc":
            ripup_indices.sort(key=lambda idx: (-key_counts[idx], idx))
        elif self.bbox_order == "count_asc":
            ripup_indices.sort(key=lambda idx: (key_counts[idx], idx))
        elif self.bbox_order == "area_asc":
            ripup_indices.sort(key=lambda idx: (self.node_areas[idx], idx))
        elif self.bbox_order == "node_id":
            # idx is the topology-based node ID assigned by get_node_id_to_name_topology().
            ripup_indices.sort()

        critical_count = len(ripup_indices)
        num_critical_ripup = max(1, int(critical_count * self.ripup_ratio))
        critical_selected = ripup_indices[:num_critical_ripup]
        self._last_candidate_count = critical_count

        selected_indices = critical_selected
        if self.bbox_order == "node_id":
            # Supplements are appended above, so sort again to preserve node-ID order
            # across every macro that will actually be re-placed.
            selected_indices.sort()
        return selected_indices

    def _check_overlap_list(self, gx, gy, w_g, h_g, placed_modules):

        for (pgx, pgy, pw_g, ph_g) in placed_modules:
            inter_w = min(gx + w_g, pgx + pw_g) - max(gx, pgx)
            inter_h = min(gy + h_g, pgy + ph_g) - max(gy, pgy)
            if inter_w > 0 and inter_h > 0:
                return True
        return False

    def _check_coarse_overlap(self, xc, yc, w_c, h_c, placed_coarse):

        for (pxc, pyc, pw_c, ph_c) in placed_coarse:
            inter_w = min(xc + w_c, pxc + pw_c) - max(xc, pxc)
            inter_h = min(yc + h_c, pyc + ph_c) - max(yc, pyc)
            if inter_w > 0 and inter_h > 0:
                return True
        return False

    def _genotype_to_phenotype(self, genotype):

        phenotype = [None] * self.num_nodes

        for idx in range(self.num_nodes):
            xc, yc, xf, yf = genotype[idx]
            span = self.node_grid_spans[idx]
            xc = max(0, min(xc, self.coarse_grid_size - span["w_c"]))
            yc = max(0, min(yc, self.coarse_grid_size - span["h_c"]))
            xf = max(0, min(xf, span["max_xf"]))
            yf = max(0, min(yf, span["max_yf"]))
            gx = xc * self.fine_grids_per_coarse_cell + xf
            gy = yc * self.fine_grids_per_coarse_cell + yf
            phenotype[idx] = (gx, gy)

        return phenotype

    def _mark_occupied(self, occupied, x, y, w, h):
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(occupied.shape[0], x + w)
        y1 = min(occupied.shape[1], y + h)
        if x0 < x1 and y0 < y1:
            occupied[x0:x1, y0:y1] = True

    def _unmark_occupied(self, occupied, x, y, w, h):
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(occupied.shape[0], x + w)
        y1 = min(occupied.shape[1], y + h)
        if x0 < x1 and y0 < y1:
            occupied[x0:x1, y0:y1] = False

    def _legal_placement_mask(self, occupied, w, h):
        if w <= 0 or h <= 0 or w > occupied.shape[0] or h > occupied.shape[1]:
            return np.zeros((0, 0), dtype=bool)
        integral = np.pad(occupied.astype(np.int32), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
        window_sum = (
            integral[w:, h:]
            - integral[:-w, h:]
            - integral[w:, :-h]
            + integral[:-w, :-h]
        )
        return window_sum == 0

    def _is_legal_position(self, occupied, x, y, w, h):
        if x < 0 or y < 0 or x + w > occupied.shape[0] or y + h > occupied.shape[1]:
            return False
        return not occupied[x:x + w, y:y + h].any()

    def _random_legal_position(self, legal_mask):
        positions = np.argwhere(legal_mask)
        if len(positions) == 0:
            return None
        pos = positions[random.randrange(len(positions))]
        return int(pos[0]), int(pos[1])

    def _first_legal_position(self, legal_mask):
        positions = np.argwhere(legal_mask)
        if len(positions) == 0:
            return None
        pos = positions[0]
        return int(pos[0]), int(pos[1])

    def _compute_cost_map(self, idx, gx_values, gy_values, phenotype):
        gx_values = np.asarray(gx_values, dtype=np.float64)
        gy_values = np.asarray(gy_values, dtype=np.float64)
        cost_map = np.zeros((len(gx_values), len(gy_values)), dtype=np.float64)
        if len(gx_values) == 0 or len(gy_values) == 0:
            return cost_map

        base_x = gx_values[:, None] * self.global_cell_width
        base_y = gy_values[None, :] * self.global_cell_height

        for net_idx in self.node_to_nets[idx]:
            nodes = self.evaluator.net_to_nodes[net_idx]
            if len(nodes) <= 1:
                continue

            candidate_offset = None
            other_x = []
            other_y = []
            for node_i, nname in enumerate(nodes):
                nidx = self.node_name_to_idx[nname]
                ox, oy = self.evaluator.net_pin_offsets[net_idx][node_i]
                if nidx == idx:
                    candidate_offset = (ox, oy)
                elif phenotype[nidx] is not None:
                    pgx, pgy = phenotype[nidx]
                    other_x.append(pgx * self.global_cell_width + ox)
                    other_y.append(pgy * self.global_cell_height + oy)

            if candidate_offset is None or not other_x:
                continue

            cand_x = base_x + candidate_offset[0]
            cand_y = base_y + candidate_offset[1]
            min_x = min(other_x)
            max_x = max(other_x)
            min_y = min(other_y)
            max_y = max(other_y)
            hpwl_x = np.maximum(max_x, cand_x) - np.minimum(min_x, cand_x)
            hpwl_y = np.maximum(max_y, cand_y) - np.minimum(min_y, cand_y)
            cost_map += (hpwl_x + hpwl_y) * self.evaluator.net_weights[net_idx]

        return cost_map

    def _select_min_cost_position(self, hpwl_increment, overlap, gx_values, gy_values):
        if hpwl_increment.shape != overlap.shape or hpwl_increment.size == 0:
            return None
        masked_cost = hpwl_increment.astype(np.float64, copy=True)
        masked_cost[overlap > overlap.min()] = np.inf
        min_cost = np.min(masked_cost)
        if not np.isfinite(min_cost):
            return None
        min_indices = np.where(masked_cost == min_cost)
        choice = random.randrange(len(min_indices[0]))
        return int(gx_values[min_indices[0][choice]]), int(gy_values[min_indices[1][choice]])

    def _compute_wire_mask(self, idx, w, h, phenotype, fine_occupied, gx_min, gx_max, gy_min, gy_max):

        w_g = int(np.ceil(w / self.global_cell_width))
        h_g = int(np.ceil(h / self.global_cell_height))
        gx_min = max(0, gx_min)
        gx_max = min(self.global_grid_size - w_g + 1, gx_max)
        gy_min = max(0, gy_min)
        gy_max = min(self.global_grid_size - h_g + 1, gy_max)
        if gx_min >= gx_max or gy_min >= gy_max:
            return {}

        gx_values = np.arange(gx_min, gx_max)
        gy_values = np.arange(gy_min, gy_max)
        if isinstance(fine_occupied, np.ndarray):
            local_occupied = fine_occupied[gx_min:gx_max + w_g - 1, gy_min:gy_max + h_g - 1]
            legal_mask = self._legal_placement_mask(local_occupied, w_g, h_g)
        else:
            legal_mask = np.ones((len(gx_values), len(gy_values)), dtype=bool)
            for ix, gx in enumerate(gx_values):
                for iy, gy in enumerate(gy_values):
                    legal_mask[ix, iy] = not self._check_overlap_list(gx, gy, w_g, h_g, fine_occupied)
        cost_map = self._compute_cost_map(idx, gx_values, gy_values, phenotype)
        return {
            (int(gx_values[ix]), int(gy_values[iy])): float(cost_map[ix, iy])
            for ix, iy in np.argwhere(legal_mask)
        }

    def _coarse_greedy_place(self, idx, phenotype, coarse_occupied):

        span = self.node_grid_spans[idx]
        w_c = span["w_c"]
        h_c = span["h_c"]
        legal_mask = self._legal_placement_mask(coarse_occupied, w_c, h_c)
        if legal_mask.size == 0 or not legal_mask.any():
            return None
        overlap = (~legal_mask).astype(np.int8)

        coarse_x = np.arange(overlap.shape[0])
        coarse_y = np.arange(overlap.shape[1])
        gx_values = coarse_x * self.fine_grids_per_coarse_cell
        gy_values = coarse_y * self.fine_grids_per_coarse_cell
        wl_incre = self._compute_cost_map(idx, gx_values, gy_values, phenotype)
        return self._select_min_cost_position(wl_incre, overlap, gx_values, gy_values)

    def _fine_greedy_place(self, idx, phenotype, fine_occupied, coarse_gx, coarse_gy):

        span = self.node_grid_spans[idx]
        w_g = span["w_g"]
        h_g = span["h_g"]
        cx = coarse_gx // self.fine_grids_per_coarse_cell
        cy = coarse_gy // self.fine_grids_per_coarse_cell

        fg_x_min = cx * self.fine_grids_per_coarse_cell
        fg_y_min = cy * self.fine_grids_per_coarse_cell
        fg_x_max = min(fg_x_min + span["max_xf"] + 1, self.global_grid_size - w_g + 1)
        fg_y_max = min(fg_y_min + span["max_yf"] + 1, self.global_grid_size - h_g + 1)
        if fg_x_min >= fg_x_max or fg_y_min >= fg_y_max:
            return (coarse_gx, coarse_gy)

        gx_values = np.arange(fg_x_min, fg_x_max)
        gy_values = np.arange(fg_y_min, fg_y_max)
        if fine_occupied is None:
            overlap = np.zeros((len(gx_values), len(gy_values)), dtype=np.int8)
        else:
            local_occupied = fine_occupied[fg_x_min:fg_x_max + w_g - 1, fg_y_min:fg_y_max + h_g - 1]
            legal_mask = self._legal_placement_mask(local_occupied, w_g, h_g)
            if legal_mask.size == 0 or not legal_mask.any():
                return (coarse_gx, coarse_gy)
            overlap = (~legal_mask).astype(np.int8)
        wl_incre = self._compute_cost_map(idx, gx_values, gy_values, phenotype)
        best_pos = self._select_min_cost_position(wl_incre, overlap, gx_values, gy_values)
        if best_pos is not None:
            return best_pos
        return (coarse_gx, coarse_gy)

    def _node_hpwl_order(self, phenotype):
        node_costs = [0.0] * self.num_nodes
        for net_idx, nodes in enumerate(self.evaluator.net_to_nodes):
            if len(nodes) <= 1:
                continue
            pins = []
            node_indices = []
            for node_i, node_name in enumerate(nodes):
                node_idx = self.node_name_to_idx[node_name]
                if phenotype[node_idx] is None:
                    continue
                gx, gy = phenotype[node_idx]
                ox, oy = self.evaluator.net_pin_offsets[net_idx][node_i]
                pins.append((gx * self.global_cell_width + ox, gy * self.global_cell_height + oy))
                node_indices.append(node_idx)
            if len(pins) <= 1:
                continue
            hpwl = _numba_hpwl(np.array(pins, dtype=np.float64)) * self.evaluator.net_weights[net_idx]
            for node_idx in node_indices:
                node_costs[node_idx] += hpwl
        return sorted(range(self.num_nodes), key=lambda idx: (-node_costs[idx], idx))

    def _stage3_critical_order(self, phenotype):
        critical_indices = list(self._bbox_critical_counts(phenotype).keys())
        if not critical_indices:
            return [], 0
        if self.stage3_order == "hpwl":
            hpwl_order = self._node_hpwl_order(phenotype)
            critical_set = set(critical_indices)
            return [idx for idx in hpwl_order if idx in critical_set], len(critical_indices)
        if self.stage3_order == "node_id":
            return sorted(critical_indices), len(critical_indices)
        random.shuffle(critical_indices)
        return critical_indices, len(critical_indices)

    def _stage3_fine_greedy_offset(self, idx, phenotype, xc, yc, current_xf, current_yf):
        span = self.node_grid_spans[idx]
        f = self.fine_grids_per_coarse_cell
        max_xf = min(f - 1, span["w_c"] * f - span["w_g"])
        max_yf = min(f - 1, span["h_c"] * f - span["h_g"])
        if max_xf < 0 or max_yf < 0:
            return current_xf, current_yf

        coarse_gx = xc * f
        coarse_gy = yc * f
        x_offsets = np.arange(f)
        y_offsets = np.arange(f)
        gx_values = coarse_gx + x_offsets
        gy_values = coarse_gy + y_offsets
        cost_map = self._compute_cost_map(idx, gx_values, gy_values, phenotype)

        legal_mask = np.zeros((f, f), dtype=bool)
        legal_mask[:max_xf + 1, :max_yf + 1] = True
        cost_map = cost_map.astype(np.float64, copy=True)
        cost_map[~legal_mask] = np.inf
        min_cost = np.min(cost_map)
        if not np.isfinite(min_cost):
            return current_xf, current_yf
        min_indices = np.where(cost_map == min_cost)
        choice = random.randrange(len(min_indices[0]))
        return int(min_indices[0][choice]), int(min_indices[1][choice])

    def _fine_only_greedy_refine(self, genotype):
        solution = copy.deepcopy(genotype)
        phenotype = [None] * self.num_nodes

        for idx in range(self.num_nodes):
            xc, yc, xf, yf = solution[idx]
            gx = xc * self.fine_grids_per_coarse_cell + xf
            gy = yc * self.fine_grids_per_coarse_cell + yf
            phenotype[idx] = (gx, gy)

        placement_order, candidate_count = self._stage3_critical_order(phenotype)

        for idx in placement_order:
            xc, yc, xf, yf = solution[idx]
            phenotype[idx] = None

            best_xf, best_yf = self._stage3_fine_greedy_offset(idx, phenotype, xc, yc, xf, yf)
            new_gx = xc * self.fine_grids_per_coarse_cell + best_xf
            new_gy = yc * self.fine_grids_per_coarse_cell + best_yf

            solution[idx] = (xc, yc, best_xf, best_yf)
            phenotype[idx] = (new_gx, new_gy)

        return solution, phenotype, candidate_count

    def _stage3_coarse_displace(self, idx, phenotype, occupied, xc, yc, xf, yf):
        """Choose the globally best legal structure-preserving displacement."""
        span = self.node_grid_spans[idx]
        f = self.fine_grids_per_coarse_cell
        xc_values = np.arange(self.coarse_grid_size - span["w_c"] + 1)
        yc_values = np.arange(self.coarse_grid_size - span["h_c"] + 1)
        gx_values = xc_values * f + xf
        gy_values = yc_values * f + yf
        cost_map = self._compute_cost_map(idx, gx_values, gy_values, phenotype)

        prefix = np.pad((occupied > 0).astype(np.int32), ((1, 0), (1, 0)))
        prefix = prefix.cumsum(axis=0).cumsum(axis=1)
        footprint_overlap = (
            prefix[span["w_c"]:, span["h_c"]:]
            - prefix[:-span["w_c"], span["h_c"]:]
            - prefix[span["w_c"]:, :-span["h_c"]]
            + prefix[:-span["w_c"], :-span["h_c"]]
        )

        candidate_x, candidate_y = np.meshgrid(xc_values, yc_values, indexing="ij")
        sweep_x1 = np.minimum(candidate_x, xc)
        sweep_x2 = np.maximum(candidate_x, xc) + 1
        sweep_y1 = np.minimum(candidate_y, yc)
        sweep_y2 = np.maximum(candidate_y, yc) + 1
        sweep_overlap = (
            prefix[sweep_x2, sweep_y2]
            - prefix[sweep_x1, sweep_y2]
            - prefix[sweep_x2, sweep_y1]
            + prefix[sweep_x1, sweep_y1]
        )
        legal_mask = (footprint_overlap == 0) & (sweep_overlap == 0)

        cost_map = cost_map.astype(np.float64, copy=True)
        cost_map[~legal_mask] = np.inf
        min_cost = np.min(cost_map)
        if not np.isfinite(min_cost):
            return xc, yc
        candidates = np.argwhere(cost_map == min_cost)
        choice = candidates[random.randrange(len(candidates))]
        return int(xc_values[choice[0]]), int(yc_values[choice[1]])

    def _coarse_displacement_refine(self, genotype):
        """Refine coarse locations using structure-preserving displacement."""
        solution = copy.deepcopy(genotype)
        phenotype = self._genotype_to_phenotype(solution)
        placement_order, candidate_count = self._stage3_critical_order(phenotype)
        occupied = np.zeros((self.coarse_grid_size, self.coarse_grid_size), dtype=np.int32)
        for node_idx in range(self.num_nodes):
            node_xc, node_yc, _, _ = solution[node_idx]
            node_span = self.node_grid_spans[node_idx]
            occupied[node_xc:node_xc + node_span["w_c"],
                     node_yc:node_yc + node_span["h_c"]] += 1

        for idx in placement_order:
            xc, yc, xf, yf = solution[idx]
            span = self.node_grid_spans[idx]
            occupied[xc:xc + span["w_c"], yc:yc + span["h_c"]] -= 1
            phenotype[idx] = None
            best_xc, best_yc = self._stage3_coarse_displace(
                idx, phenotype, occupied, xc, yc, xf, yf)
            solution[idx] = (best_xc, best_yc, xf, yf)
            phenotype[idx] = (best_xc * self.fine_grids_per_coarse_cell + xf,
                              best_yc * self.fine_grids_per_coarse_cell + yf)
            occupied[best_xc:best_xc + span["w_c"],
                     best_yc:best_yc + span["h_c"]] += 1

        return solution, phenotype, candidate_count

    def _stage3_refine(self, genotype, mode=None):
        """Apply the configured Stage 3 pass or passes to one genotype."""
        mode = self.stage3_mode if mode is None else mode
        if mode == "coarse":
            return self._coarse_displacement_refine(genotype)
        if mode == "fine":
            return self._fine_only_greedy_refine(genotype)
        if mode == "both_2":
            coarse_genotype, _, coarse_count = self._coarse_displacement_refine(genotype)
            fine_genotype, phenotype, fine_count = self._fine_only_greedy_refine(coarse_genotype)
            return fine_genotype, phenotype, max(coarse_count, fine_count)
        raise ValueError("mode must be 'coarse', 'fine', or 'both_2'")

    @staticmethod
    def _stage3_schedule(mode, iterations):
        if mode == "both_1":
            return ["coarse"] * iterations + ["fine"] * iterations
        return [mode] * iterations

    def _replacement_indices(self, ripup_indices, preserve_order=False):
        """Return unique rip-up indices in their configured replacement order."""
        ordered_indices = list(dict.fromkeys(ripup_indices))
        replace_order = self.replace_order
        if replace_order == "auto":
            # Preserve the historical behavior: bbox uses its selection order,
            # while random rip-up re-places macros in topology node-ID order.
            replace_order = "preserve" if preserve_order else "node_id"

        if replace_order == "node_id":
            ordered_indices.sort()
        elif replace_order == "random":
            random.shuffle(ordered_indices)
        elif replace_order == "area_group":
            area_groups = {}
            for idx in ordered_indices:
                area_groups.setdefault(self.node_areas[idx], []).append(idx)
            ordered_indices = []
            for area in sorted(area_groups, reverse=True):
                group = area_groups[area]
                random.shuffle(group)
                ordered_indices.extend(group)
        return ordered_indices

    def _ripup_and_replace(self, solution, ripup_indices, preserve_order=False):

        ripup_set = set(ripup_indices)

        phenotype = [None] * self.num_nodes
        coarse_occupied = np.zeros((self.coarse_grid_size, self.coarse_grid_size), dtype=bool)
        fine_occupied = np.zeros((self.global_grid_size, self.global_grid_size), dtype=bool)

        for idx in range(self.num_nodes):
            if idx in ripup_set:
                continue
            xc, yc, xf, yf = solution[idx]
            span = self.node_grid_spans[idx]
            xc = max(0, min(xc, self.coarse_grid_size - span["w_c"]))
            yc = max(0, min(yc, self.coarse_grid_size - span["h_c"]))
            xf = max(0, min(xf, span["max_xf"]))
            yf = max(0, min(yf, span["max_yf"]))
            gx = xc * self.fine_grids_per_coarse_cell + xf
            gy = yc * self.fine_grids_per_coarse_cell + yf
            solution[idx] = (xc, yc, xf, yf)

            phenotype[idx] = (gx, gy)
            self._mark_occupied(coarse_occupied, xc, yc, span["w_c"], span["h_c"])
            self._mark_occupied(fine_occupied, gx, gy, span["w_g"], span["h_g"])

        ripup_sorted = self._replacement_indices(ripup_indices, preserve_order)

        if self._verbose_mutation:
            print(f"Replacing {len(ripup_sorted)} macros...")

        cnt = 0
        for idx in ripup_sorted:
            span = self.node_grid_spans[idx]
            w_g = span["w_g"]
            h_g = span["h_g"]
            w_c = span["w_c"]
            h_c = span["h_c"]

            cnt += 1
            if self._verbose_mutation:
                print(f"placing {cnt}-th macro")
            best_pos = self._coarse_greedy_place(idx, phenotype, coarse_occupied)
            if best_pos is None:
                return None, None

            best_pos = self._fine_greedy_place(idx, phenotype, fine_occupied, best_pos[0], best_pos[1])

            phenotype[idx] = best_pos

            new_xc = best_pos[0] // self.fine_grids_per_coarse_cell
            new_yc = best_pos[1] // self.fine_grids_per_coarse_cell
            self._mark_occupied(coarse_occupied, new_xc, new_yc, w_c, h_c)
            self._mark_occupied(fine_occupied, best_pos[0], best_pos[1], w_g, h_g)

            xc = best_pos[0] // self.fine_grids_per_coarse_cell
            yc = best_pos[1] // self.fine_grids_per_coarse_cell
            xf = best_pos[0] % self.fine_grids_per_coarse_cell
            yf = best_pos[1] % self.fine_grids_per_coarse_cell
            xc = max(0, min(self.coarse_grid_size - span["w_c"], xc))
            yc = max(0, min(self.coarse_grid_size - span["h_c"], yc))
            xf = max(0, min(span["max_xf"], xf))
            yf = max(0, min(span["max_yf"], yf))
            solution[idx] = (xc, yc, xf, yf)

        return solution, phenotype

    def _coarse_to_fine_map(self, solution):

        phenotype = [None] * self.num_nodes
        coarse_occupied = np.zeros((self.coarse_grid_size, self.coarse_grid_size), dtype=bool)

        for idx in range(self.num_nodes):
            span = self.node_grid_spans[idx]
            num_coarse_w = span["w_c"]
            num_coarse_h = span["h_c"]

            if idx not in solution:
                legal_mask = self._legal_placement_mask(coarse_occupied, num_coarse_w, num_coarse_h)
                legal_pos = self._first_legal_position(legal_mask)
                if legal_pos is None:
                    raise RuntimeError(f"No legal coarse position found for {self.node_id_to_name[idx]}")
                xc, yc = legal_pos
                xf = random.randint(0, span["max_xf"])
                yf = random.randint(0, span["max_yf"])
                solution[idx] = (xc, yc, xf, yf)

            xc, yc, xf, yf = solution[idx]

            xc = max(0, min(xc, self.coarse_grid_size - num_coarse_w))
            yc = max(0, min(yc, self.coarse_grid_size - num_coarse_h))

            xf = max(0, min(xf, span["max_xf"]))
            yf = max(0, min(yf, span["max_yf"]))

            if not self._is_legal_position(coarse_occupied, xc, yc, num_coarse_w, num_coarse_h):
                legal_mask = self._legal_placement_mask(coarse_occupied, num_coarse_w, num_coarse_h)
                legal_pos = self._first_legal_position(legal_mask)
                if legal_pos is None:
                    raise RuntimeError(f"No legal coarse position found for {self.node_id_to_name[idx]}")
                xc, yc = legal_pos

            gx = xc * self.fine_grids_per_coarse_cell + xf
            gy = yc * self.fine_grids_per_coarse_cell + yf

            phenotype[idx] = (gx, gy)
            self._mark_occupied(coarse_occupied, xc, yc, num_coarse_w, num_coarse_h)

            solution[idx] = (xc, yc, xf, yf)

        return phenotype

    def _random_solution(self):

        max_init_attempts = 50 if self.random_init_order else 1
        last_failed_node = None

        for _ in range(max_init_attempts):
            genotype = {}
            phenotype = [None] * self.num_nodes
            coarse_occupied = np.zeros((self.coarse_grid_size, self.coarse_grid_size), dtype=bool)
            fine_occupied = np.zeros((self.global_grid_size, self.global_grid_size), dtype=bool)
            placement_order = list(range(self.num_nodes))
            if self.random_init_order:
                random.shuffle(placement_order)

            failed = False
            for idx in placement_order:
                span = self.node_grid_spans[idx]
                best_pos = self._coarse_greedy_place(idx, phenotype, coarse_occupied)
                if best_pos is None:
                    last_failed_node = self.node_id_to_name[idx]
                    failed = True
                    break
                best_pos = self._fine_greedy_place(idx, phenotype, fine_occupied, best_pos[0], best_pos[1])

                gx, gy = best_pos
                xc = gx // self.fine_grids_per_coarse_cell
                yc = gy // self.fine_grids_per_coarse_cell
                xf = gx % self.fine_grids_per_coarse_cell
                yf = gy % self.fine_grids_per_coarse_cell
                xc = max(0, min(self.coarse_grid_size - span["w_c"], xc))
                yc = max(0, min(self.coarse_grid_size - span["h_c"], yc))
                xf = max(0, min(span["max_xf"], xf))
                yf = max(0, min(span["max_yf"], yf))
                gx = xc * self.fine_grids_per_coarse_cell + xf
                gy = yc * self.fine_grids_per_coarse_cell + yf

                genotype[idx] = (xc, yc, xf, yf)
                phenotype[idx] = (gx, gy)
                self._mark_occupied(coarse_occupied, xc, yc, span["w_c"], span["h_c"])
                self._mark_occupied(fine_occupied, gx, gy, span["w_g"], span["h_g"])

            if not failed:
                print(f"greedy solution finished, keys count: {len(genotype)}")
                return genotype, phenotype

        print(f"No legal coarse position found for {last_failed_node} after {max_init_attempts} initialization attempts")
        return {}, []

    def _evaluate(self, genotype, phenotype=None):

        if phenotype is None:
            phenotype = self._genotype_to_phenotype(genotype)

        hpwl = 0.0
        for net_idx, nodes in enumerate(self.evaluator.net_to_nodes):
            pins = []
            for node_i, node_name in enumerate(nodes):
                node_idx = self.node_name_to_idx[node_name]
                gx, gy = phenotype[node_idx]
                px = gx * self.global_cell_width
                py = gy * self.global_cell_height
                ox, oy = self.evaluator.net_pin_offsets[net_idx][node_i]
                pins.append((px + ox, py + oy))
            if len(pins) <= 1:
                continue
            pins_arr = np.array(pins, dtype=np.float64)
            hpwl += _numba_hpwl(pins_arr) * self.evaluator.net_weights[net_idx]

        overlap = 0.0
        placed_modules = []
        for i in range(self.num_nodes):
            name_i = self.node_id_to_name[i]
            ni = self.node_info[name_i]
            gxi, gyi = phenotype[i]
            pxi = gxi * self.global_cell_width
            pyi = gyi * self.global_cell_height
            wi = ni["x"]
            hi = ni["y"]

            for (pxj, pyj, wj, hj) in placed_modules:
                inter_w = min(pxi + wi, pxj + wj) - max(pxi, pxj)
                inter_h = min(pyi + hi, pyj + hj) - max(pyi, pyj)
                if inter_w > 0 and inter_h > 0:
                    overlap += inter_w * inter_h
            placed_modules.append((pxi, pyi, wi, hi))

        congestion = self.evaluator.compute_rudy_congestion_from_phenotype(phenotype)

        cost = hpwl + self.lambda_cong * congestion + self.lambda_overlap * overlap
        return -cost, hpwl, overlap, congestion, phenotype

    def _evaluate_from_phenotype(self, phenotype):

        hpwl = 0.0
        for net_idx, nodes in enumerate(self.evaluator.net_to_nodes):
            pins = []
            for node_i, node_name in enumerate(nodes):
                node_idx = self.node_name_to_idx[node_name]
                gx, gy = phenotype[node_idx]
                px = gx * self.global_cell_width
                py = gy * self.global_cell_height
                ox, oy = self.evaluator.net_pin_offsets[net_idx][node_i]
                pins.append((px + ox, py + oy))
            if len(pins) <= 1:
                continue
            pins_arr = np.array(pins, dtype=np.float64)
            hpwl += _numba_hpwl(pins_arr) * self.evaluator.net_weights[net_idx]

        overlap = 0.0
        placed_modules = []
        for i in range(self.num_nodes):
            name_i = self.node_id_to_name[i]
            ni = self.node_info[name_i]
            gxi, gyi = phenotype[i]
            pxi = gxi * self.global_cell_width
            pyi = gyi * self.global_cell_height
            wi = ni["x"]
            hi = ni["y"]

            for (pxj, pyj, wj, hj) in placed_modules:
                inter_w = min(pxi + wi, pxj + wj) - max(pxi, pxj)
                inter_h = min(pyi + hi, pyj + hj) - max(pyi, pyj)
                if inter_w > 0 and inter_h > 0:
                    overlap += inter_w * inter_h
            placed_modules.append((pxi, pyi, wi, hi))

        congestion = self.evaluator.compute_rudy_congestion_from_phenotype(phenotype)

        cost = hpwl + self.lambda_cong * congestion + self.lambda_overlap * overlap
        return -cost, hpwl, overlap, congestion

    def _tournament_select(self, population):
        contestants = random.sample(population, min(self.tournament_size, len(population)))
        return max(contestants, key=lambda ind: ind.fitness)

    # @staticmethod
    # def _decile_random_ripup_ratio(iteration, total_iterations):
    #     if total_iterations <= 0:
    #         raise ValueError("total_iterations must be positive")
    #     decile = min(9, iteration * 10 // total_iterations)

    #     reverse_decile = 9 - decile
    #     lower = reverse_decile * 0.05
    #     upper = (reverse_decile + 1) * 0.05
    #     return upper - (upper - lower) * random.random()

    @staticmethod
    def _decile_random_ripup_ratio(iteration, total_iterations):
        if total_iterations <= 0:
            raise ValueError("total_iterations must be positive")

        num_bins = 5
        idx = min(num_bins - 1, iteration * num_bins // total_iterations)
        reverse_idx = num_bins - 1 - idx   # 0→4, 1→3, 2→2, 3→1, 4→0

        width = 0.5 / num_bins
        lower = reverse_idx * width
        upper = (reverse_idx + 1) * width
        return upper - (upper - lower) * random.random()
    # def _decile_random_ripup_ratio(iteration, total_iterations):
    #     """Sample from ten descending 0.1-wide bands over a stage."""
    #     if total_iterations <= 0:
    #         raise ValueError("total_iterations must be positive")
    #     decile = min(9, iteration * 10 // total_iterations)
    #     lower = (9 - decile) / 10.0
    #     upper = (10 - decile) / 10.0
    #     return random.uniform(lower, upper)

    def _mutate(self, genotype):

        original_genotype = copy.deepcopy(genotype)
        new_genotype = copy.deepcopy(genotype)

        if self.ripup_strategy == "bbox":
            candidate_count = 0
            ripup_indices = self._select_bbox_ripup_indices(new_genotype)
            candidate_count = getattr(self, "_last_candidate_count", len(ripup_indices))
            new_genotype, phenotype = self._ripup_and_replace(
                new_genotype, ripup_indices, preserve_order=True)
        else:
            candidate_count = self.num_nodes
            num_ripup = max(1, int(self.num_nodes * self.ripup_ratio))
            ripup_indices = random.sample(range(self.num_nodes), num_ripup)
            new_genotype, phenotype = self._ripup_and_replace(new_genotype, ripup_indices)

        if phenotype is None:
            new_genotype = original_genotype
            phenotype = self._genotype_to_phenotype(new_genotype)

        fitness, hpwl, overlap, congestion = self._evaluate_from_phenotype(phenotype)

        return new_genotype, fitness, hpwl, overlap, congestion, phenotype, len(ripup_indices), candidate_count

    def _evolve_population(self, population, best_ever, best_hpwl_individual, hpwl_log_path, save_path, verbose, phase_prefix=""):
        stage1_phase = f"{phase_prefix}_stage1" if phase_prefix else "stage1"
        stage2_phase = f"{phase_prefix}_stage2" if phase_prefix else "stage2"
        stage3_phase = f"{phase_prefix}_stage3" if phase_prefix else "stage3"

        base_ripup_ratio = self.ripup_ratio
        stage1_ripup_ratio = base_ripup_ratio

        # Stage 1: Global Exploration (ripup & replace with configured ratio)
        if verbose:
            if self.ripup_ratio_strategy == "random_stage":
                schedule = "random [0.4,0.7]"
            elif self.ripup_ratio_strategy == "decile_random":
                schedule = "decile_random [0.9,1.0] -> [0.0,0.1]"
            else:
                schedule = f"fixed {stage1_ripup_ratio}"
            print(f"\n=== Stage 1: Global Exploration ({self.stage1_iters} iters, ripup_ratio={schedule}) ===")
        for iteration in range(self.stage1_iters):
            parent = self._tournament_select(population)

            old_ripup = self.ripup_ratio
            if self.ripup_ratio_strategy == "random_stage":
                current_ripup_ratio = random.uniform(0.4, 0.7)
            elif self.ripup_ratio_strategy == "decile_random":
                current_ripup_ratio = self._decile_random_ripup_ratio(
                    iteration, self.stage1_iters)
            else:
                current_ripup_ratio = stage1_ripup_ratio
            self.ripup_ratio = current_ripup_ratio
            new_genotype, fitness, hpwl, overlap, congestion, phenotype, ripup_count, candidate_count = self._mutate(copy.deepcopy(parent.genotype))
            self.ripup_ratio = old_ripup

            offspring = Individual(new_genotype, phenotype, fitness, hpwl, overlap, congestion)
            is_valid = self.validate_individual(offspring).is_valid

            improved = False
            saved_path = ""
            if is_valid:
                population.append(offspring)
                population.sort(key=lambda ind: ind.fitness, reverse=True)
                population = population[:self.pop_size]

                if offspring.fitness > best_ever.fitness:
                    best_ever = copy.deepcopy(offspring)
                if offspring.hpwl < best_hpwl_individual.hpwl:
                    best_hpwl_individual = copy.deepcopy(offspring)
                    improved = True
                    if save_path:
                        saved_path = self._save_best_individual(best_hpwl_individual, save_path, self.placedb.benchmark)
                        print(f"  New best HPWL found! HPWL={best_hpwl_individual.hpwl:.2e}, saved to: {saved_path}")

            self._append_hpwl_log(
                hpwl_log_path,
                stage1_phase,
                iteration + 1,
                offspring,
                best_hpwl_individual.hpwl,
                ripup_count=ripup_count,
                improved=improved,
                saved_path=saved_path,
                candidate_count=candidate_count,
                ripup_ratio=current_ripup_ratio,
            )

            if verbose and (iteration + 1) % max(1, self.stage1_iters // 10) == 0:
                print(f"  Iter {iteration+1}/{self.stage1_iters}: best_fitness={best_ever.fitness:.2e}, best_hpwl={best_hpwl_individual.hpwl:.2e}, overlap={best_ever.overlap:.2e}")

        stage2_ripup_ratio = max(0.05, base_ripup_ratio * 0.5)

        # Stage 2: Local Refinement (ripup & replace with smaller ratio)
        if verbose:
            stage2_schedule = "random [0.1,0.4]" if self.ripup_ratio_strategy == "random_stage" else f"{stage2_ripup_ratio}"
            print(f"\n=== Stage 2: Local Refinement ({self.stage2_iters} iters, ripup_ratio={stage2_schedule}) ===")
        for iteration in range(self.stage2_iters):
            parent = self._tournament_select(population)

            old_ripup = self.ripup_ratio
            current_ripup_ratio = random.uniform(0.1, 0.4) if self.ripup_ratio_strategy == "random_stage" else stage2_ripup_ratio
            self.ripup_ratio = current_ripup_ratio
            new_genotype, fitness, hpwl, overlap, congestion, phenotype, ripup_count, candidate_count = self._mutate(copy.deepcopy(parent.genotype))
            self.ripup_ratio = old_ripup

            offspring = Individual(new_genotype, phenotype, fitness, hpwl, overlap, congestion)
            is_valid = self.validate_individual(offspring).is_valid

            improved = False
            saved_path = ""
            if is_valid:
                population.append(offspring)
                population.sort(key=lambda ind: ind.fitness, reverse=True)
                population = population[:self.pop_size]

                if offspring.fitness > best_ever.fitness:
                    best_ever = copy.deepcopy(offspring)
                if offspring.hpwl < best_hpwl_individual.hpwl:
                    best_hpwl_individual = copy.deepcopy(offspring)
                    improved = True
                    if save_path:
                        saved_path = self._save_best_individual(best_hpwl_individual, save_path, self.placedb.benchmark)
                        print(f"  New best HPWL found! HPWL={best_hpwl_individual.hpwl:.2e}, saved to: {saved_path}")

            self._append_hpwl_log(
                hpwl_log_path,
                stage2_phase,
                iteration + 1,
                offspring,
                best_hpwl_individual.hpwl,
                ripup_count=ripup_count,
                improved=improved,
                saved_path=saved_path,
                candidate_count=candidate_count,
                ripup_ratio=current_ripup_ratio,
            )

            if verbose and (iteration + 1) % max(1, self.stage2_iters // 10) == 0:
                print(f"  Iter {iteration+1}/{self.stage2_iters}: best_fitness={best_ever.fitness:.2e}, best_hpwl={best_hpwl_individual.hpwl:.2e}, overlap={best_ever.overlap:.2e}")

        stage3_schedule = self._stage3_schedule(self.stage3_mode, self.stage3_iters)
        if verbose:
            stage3_names = {
                "coarse": "Structure-Preserving Coarse Displacement",
                "fine": "Fine Offset Refinement",
                "both_1": "Coarse Displacement, then Fine Offset Refinement",
                "both_2": "Coarse Displacement + Fine Offset Refinement per Iteration",
            }
            print(f"\n=== Stage 3: {stage3_names[self.stage3_mode]} ({len(stage3_schedule)} iters) ===")
        for iteration, mode in enumerate(stage3_schedule):
            parent = self._tournament_select(population)
            new_genotype, phenotype, candidate_count = self._stage3_refine(
                copy.deepcopy(parent.genotype), mode=mode)
            fitness, hpwl, overlap, congestion, _ = self._evaluate(new_genotype, phenotype)
            offspring = Individual(new_genotype, phenotype, fitness, hpwl, overlap, congestion)
            is_valid = self.validate_individual(offspring).is_valid

            improved = False
            saved_path = ""
            if is_valid:
                population.append(offspring)
                population.sort(key=lambda ind: ind.fitness, reverse=True)
                population = population[:self.pop_size]

                if offspring.fitness > best_ever.fitness:
                    best_ever = copy.deepcopy(offspring)
                if offspring.hpwl < best_hpwl_individual.hpwl:
                    best_hpwl_individual = copy.deepcopy(offspring)
                    improved = True
                    if save_path:
                        saved_path = self._save_best_individual(best_hpwl_individual, save_path, self.placedb.benchmark)
                        print(f"  New best HPWL found! HPWL={best_hpwl_individual.hpwl:.2e}, saved to: {saved_path}")

            self._append_hpwl_log(
                hpwl_log_path,
                stage3_phase,
                iteration + 1,
                offspring,
                best_hpwl_individual.hpwl,
                ripup_count=candidate_count,
                improved=improved,
                saved_path=saved_path,
                candidate_count=candidate_count,
                ripup_ratio=1.0,
            )

            if verbose and (iteration + 1) % max(1, len(stage3_schedule) // 10) == 0:
                print(f"  Iter {iteration+1}/{len(stage3_schedule)}: best_fitness={best_ever.fitness:.2e}, best_hpwl={best_hpwl_individual.hpwl:.2e}, overlap={best_ever.overlap:.2e}")

        return population, best_ever, best_hpwl_individual

    def run(self, verbose=True, save_path=None):
        population = []
        hpwl_log_path = self._hpwl_log_path(save_path) if save_path else None
        if hpwl_log_path:
            self._init_hpwl_log(hpwl_log_path)
        if verbose:
            print("Initializing population...")

        best_ever = None
        best_hpwl_individual = None
        for i in range(self.pop_size):
            genotype, phenotype = self._random_solution()
            if not genotype or not phenotype:
                if verbose:
                    print(f"  Init {i}: skipped empty solution")
                continue

            fitness, hpwl, overlap, congestion, _ = self._evaluate(genotype, phenotype)
            individual = Individual(genotype, phenotype, fitness, hpwl, overlap, congestion)
            is_valid = self.validate_individual(individual).is_valid
            improved = False
            saved_path = ""

            if not is_valid:
                if verbose:
                    print(f"  Init {i}: skipped invalid solution, hpwl={hpwl:.2e}, overlap={overlap:.2e}")
                continue

            population.append(individual)
            if best_ever is None or fitness > best_ever.fitness:
                best_ever = copy.deepcopy(individual)
                if verbose:
                    print(f"  Better solution found during init: fitness={fitness:.2e}, hpwl={hpwl:.2e}, overlap={overlap:.2e}")
            if best_hpwl_individual is None or hpwl < best_hpwl_individual.hpwl:
                best_hpwl_individual = copy.deepcopy(individual)
                improved = True
                if save_path and len(population) > 1:
                    saved_path = self._save_best_individual(best_hpwl_individual, save_path, self.placedb.benchmark)
                    if verbose:
                        print(f"  New best HPWL during init: HPWL={hpwl:.2e}, saved to: {saved_path}")

            phase = "init_base" if len(population) == 1 else "init"
            self._append_hpwl_log(
                hpwl_log_path,
                phase,
                i,
                individual,
                best_hpwl_individual.hpwl,
                ripup_count=0,
                improved=improved,
                saved_path=saved_path,
            )

        if not population:
            raise RuntimeError("No valid initial individuals were generated")

        population.sort(key=lambda ind: ind.fitness, reverse=True)

        if verbose:
            print(f"\nInitial best: fitness={best_ever.fitness:.2e}, hpwl={best_ever.hpwl:.2e}, overlap={best_ever.overlap:.2e}")
            print(f"Initial best HPWL: hpwl={best_hpwl_individual.hpwl:.2e}, fitness={best_hpwl_individual.fitness:.2e}")

        if save_path:
            saved_path = self._save_best_individual(best_hpwl_individual, save_path, self.placedb.benchmark)
            self._append_hpwl_log(hpwl_log_path, "initial_best_hpwl", 0, best_hpwl_individual, best_hpwl_individual.hpwl, improved=True, saved_path=saved_path)
            print(f"Initial best-HPWL placement saved to: {saved_path}")

        population, best_ever, best_hpwl_individual = self._evolve_population(
            population,
            best_ever,
            best_hpwl_individual,
            hpwl_log_path,
            save_path,
            verbose,
        )

        return best_hpwl_individual

    def run_stage3(self, genotype, iterations=None, verbose=True, save_path=None,
                   mode="fine"):
        """Run coarse displacement or fine-offset Stage 3 on one placement.

        ``genotype`` must use the engine's ``{node_id: (xc, yc, xf, yf)}``
        representation.  This method intentionally does not perform Stage 1
        or Stage 2, and is useful for refining a placement loaded from a
        Bookshelf ``.pl`` file.
        """
        if iterations is None:
            iterations = self.stage3_iters
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        if mode not in {"coarse", "fine", "both_1", "both_2"}:
            raise ValueError("mode must be 'coarse', 'fine', 'both_1', or 'both_2'")
        if not isinstance(genotype, dict) or len(genotype) != self.num_nodes:
            raise ValueError(f"genotype must contain all {self.num_nodes} node ids")

        genotype = copy.deepcopy(genotype)
        phenotype = self._genotype_to_phenotype(genotype)
        fitness, hpwl, overlap, congestion, phenotype = self._evaluate(genotype, phenotype)
        best = Individual(genotype, phenotype, fitness, hpwl, overlap, congestion)
        validation = self.validate_individual(best)
        if not validation.is_valid:
            raise ValueError("input placement is not legal: " + "; ".join(validation.errors[:5]))

        hpwl_log_path = self._hpwl_log_path(save_path) if save_path else None
        if hpwl_log_path:
            self._init_hpwl_log(hpwl_log_path)
        stage3_schedule = self._stage3_schedule(mode, iterations)
        if verbose:
            stage_names = {
                "coarse": "Structure-Preserving Coarse Displacement",
                "fine": "Fine Offset Refinement",
                "both_1": "Coarse Displacement, then Fine Offset Refinement",
                "both_2": "Coarse Displacement + Fine Offset Refinement per Iteration",
            }
            stage_name = stage_names[mode]
            print(f"=== Stage 3: {stage_name} ({len(stage3_schedule)} iters) ===")
            print(f"Initial HPWL: {best.hpwl:.6e}")
        if save_path:
            saved = self._save_best_individual(best, save_path, self.placedb.benchmark)
            self._append_hpwl_log(hpwl_log_path, "stage3_initial", 0, best, best.hpwl,
                                  improved=True, saved_path=saved, candidate_count=self.num_nodes,
                                  ripup_count=self.num_nodes, ripup_ratio=1.0)

        for iteration, pass_mode in enumerate(stage3_schedule):
            new_genotype, new_phenotype, candidate_count = self._stage3_refine(
                copy.deepcopy(best.genotype), mode=pass_mode)
            fitness, hpwl, overlap, congestion, new_phenotype = self._evaluate(new_genotype, new_phenotype)
            candidate = Individual(new_genotype, new_phenotype, fitness, hpwl, overlap, congestion)
            if not self.validate_individual(candidate).is_valid:
                continue
            improved = candidate.hpwl < best.hpwl
            saved = ""
            if improved:
                best = candidate
                if save_path:
                    saved = self._save_best_individual(best, save_path, self.placedb.benchmark)
            self._append_hpwl_log(hpwl_log_path, f"stage3_{mode}", iteration + 1, candidate, best.hpwl,
                                  improved=improved, saved_path=saved, candidate_count=candidate_count,
                                  ripup_count=candidate_count, ripup_ratio=1.0)
            if verbose and (iteration + 1) % max(1, len(stage3_schedule) // 10) == 0:
                print(f"  Iter {iteration + 1}/{len(stage3_schedule)}: best_hpwl={best.hpwl:.6e}")
        if save_path:
            self._save_best_individual(best, save_path, self.placedb.benchmark)
        return best

    def _hpwl_log_path(self, output_path):
        dir_name = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        stem, _ = os.path.splitext(base_name)
        log_name = f"{stem}_hpwl_history.txt"
        return os.path.join(dir_name, log_name) if dir_name else log_name

    def _init_hpwl_log(self, log_path):
        dir_name = os.path.dirname(log_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(log_path, 'w') as f:
            f.write("phase\titeration\tcandidate_count\tripup_count\tcandidate_hpwl\tbest_hpwl\tfitness\toverlap\tcongestion\timproved\tsaved_path\tripup_ratio\n")

    def _append_hpwl_log(
        self,
        log_path,
        phase,
        iteration,
        individual,
        best_hpwl,
        ripup_count=0,
        improved=False,
        saved_path="",
        candidate_count=0,
        ripup_ratio=0.0,
    ):
        if log_path is None:
            return
        record = {
            "phase": phase,
            "iteration": iteration,
            "candidate_count": candidate_count,
            "ripup_count": ripup_count,
            "candidate_hpwl": individual.hpwl,
            "best_hpwl": best_hpwl,
            "fitness": individual.fitness,
            "overlap": individual.overlap,
            "congestion": individual.congestion,
            "improved": int(improved),
            "saved_path": saved_path,
            "ripup_ratio": ripup_ratio,
        }
        if isinstance(log_path, list):
            log_path.append(record)
            return
        self._write_hpwl_record(log_path, record)

    def _write_hpwl_record(self, log_path, record):
        with open(log_path, 'a') as f:
            f.write(
                f"{record['phase']}\t{record['iteration']}\t{record['candidate_count']}\t{record['ripup_count']}\t"
                f"{record['candidate_hpwl']:.10e}\t{record['best_hpwl']:.10e}\t"
                f"{record['fitness']:.10e}\t{record['overlap']:.10e}\t{record['congestion']:.10e}\t"
                f"{record['improved']}\t{record['saved_path']}\t{record['ripup_ratio']:.10g}\n"
            )

    def _save_best_individual(self, individual, output_path, benchmark_name, include_hpwl=True):
        dir_name = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        if include_hpwl:
            hpwl_int = int(individual.hpwl)
            stem, ext = os.path.splitext(base_name)
            base_name = f"{stem}_{hpwl_int}{ext}"
        new_output_path = os.path.join(dir_name, base_name) if dir_name else base_name

        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(new_output_path, 'w') as f:
            f.write("UCLA pl 1.0\n")
            f.write(f"\n# Created by C2FPlace-EA\n")
            f.write(f"# Benchmark: {benchmark_name}\n")
            f.write(f"# HPWL: {individual.hpwl:.2e}\n")
            f.write(f"# Overlap: {individual.overlap:.2e}\n")
            f.write(f"# Congestion: {individual.congestion:.2e}\n\n")

            for idx in range(self.num_nodes):
                node_name = self.node_id_to_name[idx]
                gx, gy = individual.phenotype[idx]
                x = gx * self.global_cell_width
                y = gy * self.global_cell_height
                f.write(f"{node_name}\t{int(x)}\t{int(y)}\t: N\n")

        return new_output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="adaptec1")
    parser.add_argument("--coarse_grid", type=int, default=224, help="Coarse grid size")
    parser.add_argument("--pop_size", type=int, default=1)
    parser.add_argument("--stage1_iters", type=int, default=5000)
    parser.add_argument("--stage2_iters", type=int, default=0)
    parser.add_argument("--stage3_iters", type=int, default=0)
    parser.add_argument("--stage3_mode", type=str, default="both_2", choices=["coarse", "fine", "both_1", "both_2"])
    parser.add_argument("--stage3_order", type=str, default="node_id", choices=["hpwl", "random", "node_id"])
    parser.add_argument("--ripup_ratio", type=float, default=0.2)
    parser.add_argument("--ripup_strategy", type=str, default="random", choices=["random", "bbox"],
                        help="Rip-up selection strategy: random or bbox")
    parser.add_argument("--bbox_order", type=str, default="node_id", choices=["random", "count_desc", "count_asc", "area_asc", "node_id"],
                        help="Ranking used to select bbox rip-up macros; node_id follows topology order")
    parser.add_argument("--replace_order", type=str, default="node_id", choices=["auto", "preserve", "node_id", "random", "area_group"],
                        help="Order for re-placing selected macros; auto preserves legacy strategy-specific behavior")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_init_order", action="store_true",
                        help="Shuffle macro placement order for each greedy initialization")
    parser.add_argument("--ripup_ratio_strategy", type=str, default="fixed",
                        choices=["fixed", "random_stage", "decile_random"],
                        help="Rip-up ratio schedule; decile_random uses descending random bands during Stage 1")
    parser.add_argument("--tournament_size", type=int, default=5,
                            help="Size of the tournament for selection")
    parser.add_argument("--output", type=str, default=None, help="Output .pl file path")
    args = parser.parse_args()

    print(f"Loading benchmark: {args.benchmark}")
    placedb = PlaceDB(args.benchmark)
    placedb.debug_str()

    if args.output is None:
        result_tag = args.ripup_strategy
        if args.ripup_strategy == "bbox":
            result_tag = f"bbox_{args.bbox_order}"
        if args.ripup_ratio_strategy == "random_stage":
            result_tag = f"{result_tag}_randratio"
        elif args.ripup_ratio_strategy == "decile_random":
            result_tag = f"{result_tag}_decilerand"
        if args.replace_order != "auto":
            result_tag = f"{result_tag}_repl{args.replace_order}"
        args.output = f"results/{args.benchmark}/{result_tag}/{args.benchmark}_c2fplace.pl"

    print(f"\nRunning C2FPlace-EA ...")
    ea = C2FPlaceEA(
        placedb,
        pop_size=args.pop_size,
        coarse_grid_size=args.coarse_grid,
        stage1_iters=args.stage1_iters,
        stage2_iters=args.stage2_iters,
        stage3_iters=args.stage3_iters,
        stage3_mode=args.stage3_mode,
        stage3_order=args.stage3_order,
        ripup_ratio=args.ripup_ratio,
        ripup_strategy=args.ripup_strategy,
        bbox_order=args.bbox_order,
        replace_order=args.replace_order,
        random_init_order=args.random_init_order,
        ripup_ratio_strategy=args.ripup_ratio_strategy,
        seed=args.seed
    )

    best = ea.run(verbose=True, save_path=args.output)
    print(f"\n=== Final Result ===")
    print(f"HPWL: {best.hpwl:.2e}")
    print(f"Overlap: {best.overlap:.2e}")
    print(f"Congestion: {best.congestion:.2e}")
    print(f"Fitness: {best.fitness:.2e}")

    validation = ea.validate_individual(best)
    print(f"Valid: {validation.is_valid}")
    if validation.errors:
        print("Validation errors:")
        for error in validation.errors[:10]:
            print(f"  - {error}")
        if len(validation.errors) > 10:
            print(f"  - ... {len(validation.errors) - 10} more")

    saved_path = ea._save_best_individual(best, args.output, args.benchmark, include_hpwl=True)
    print(f"Saved to: {saved_path}")

    print(f"\nBest placement summary:")
    print(f"  Total nodes: {ea.num_nodes}")
    print(f"  Coarse grid: {ea.coarse_grid_size}x{ea.coarse_grid_size}")
    print(f"  Fine grids per coarse cell: {ea.fine_grids_per_coarse_cell}x{ea.fine_grids_per_coarse_cell}")
    print(f"  Global grid: {ea.global_grid_size}x{ea.global_grid_size}")
