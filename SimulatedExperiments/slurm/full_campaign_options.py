#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# Slurm-friendly top matter
# =========================
import os, sys, copy, time, json, pickle, ast

# Honor Slurm CPU allocation BEFORE importing numpy/scipy/sklearn
threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1")))
for _var in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"
]:
    os.environ[_var] = str(threads)

# Headless plotting on compute nodes
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import xarray as xr
import glob as glob
from math import pi
from zipfile import ZipFile, ZIP_DEFLATED
from typing import Tuple
from pathlib import Path
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from tqdm import trange, tqdm  # plain tqdm for headless logs
from sklearn.metrics import silhouette_samples, silhouette_score, classification_report, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
from scipy.signal import argrelmin, argrelmax, argrelextrema
from scipy.stats import entropy
from scipy.interpolate import griddata, RBFInterpolator, LinearNDInterpolator
import scipy.ndimage as sndi
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.spatial import distance_matrix, ConvexHull
from functools import lru_cache
from joblib import Parallel, delayed
import cv2
import argparse

mpl.rcParams['mathtext.default'] = 'regular'

def _runtime_sanity():
    # What the script decided to use
    print(f"[sanity] threads variable = {threads}")
    print(f"[sanity] SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')}")
    print(f"[sanity] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"[sanity] OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    print(f"[sanity] MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")

    # What Linux cgroups/Slurm allow this process to touch
    try:
        aff = os.sched_getaffinity(0)
        print(f"[sanity] CPU affinity: {len(aff)} CPUs visible to this process")
    except Exception as e:
        print(f"[sanity] CPU affinity check failed: {e}")

    # What threadpools BLAS/OpenMP have actually configured
    try:
        from threadpoolctl import threadpool_info
        info = threadpool_info()
        print("[sanity] BLAS/OpenMP threadpools detected:")
        for p in info:
            print(f"  - {p.get('internal_api') or p.get('user_api')} "
                  f"{p.get('filepath','')} num_threads={p.get('num_threads')}")
    except Exception as e:
        print(f"[sanity] threadpoolctl not available or no pools yet: {e}")


# =========================
# Paths (override via env)
# =========================
# Default Orion paths; can be overridden by environment variables below.
path = os.environ.get("SIM_DATA_DIR", "/nsls2/users/acorrao/GitHub/pub-Corrao_2025_09/Data/simulatedwafer_datasets")
simexp_path = os.environ.get("SIM_CODE_DIR", '/nsls2/users/acorrao/GitHub/pub-Corrao_2025_09/SimulatedExperiments')
fig_path = os.environ.get("SIM_FIG_DIR", '/nsls2/users/acorrao/GitHub/pub-Corrao_2025_09/SimulatedExperiments/figures')
pkl_path = os.environ.get("SIM_PKL_DIR", '/nsls2/users/acorrao/GitHub/pub-Corrao_2025_09/SimulatedExperiments/pkls')
png_path = os.environ.get("SIM_PNG_DIR", '/nsls2/users/acorrao/GitHub/pub-Corrao_2025_09/SimulatedExperiments/pngs')
movie_path = os.environ.get("SIM_MOVIE_DIR", '/nsls2/users/acorrao/GitHub/pub-Corrao_2025_09/SimulatedExperiments/movies')
snapshot_path = os.environ.get("SIM_SNAPSHOT_DIR", '/nsls2/users/acorrao/GitHub/pub-Corrao_2025_09/SimulatedExperiments/pngs/campaign_snapshots')

# Ensure output dirs exist
for d in [fig_path, pkl_path, png_path, movie_path, snapshot_path]:
    os.makedirs(d, exist_ok=True)

# If you ever need to run on your workstation, you can comment the above and re-enable local defaults.

# =========================
# Dataset file names
# =========================
# complex wafer
complex_zipfile = 'ds_AlLiFe_complex_21Sep2024_12-04-04.zip'
complex_file = 'ds_AlLiFe_complex_21Sep2024_12-04-04.nc'

# simple L/R wafer
simpleLR_zipfile = 'ds_simpleLR_22Aug2024_13-12-03.zip'
simpleLR_file = 'ds_simpleLR_22Aug2024_13-12-03.nc'

# Dan sam 1 (square)
dansam1_square_zipfile = 'dansam1_square_17Sep2024_12-50-00.zip'
dansam1_square_file = 'dansam1_square_17Sep2024_12-50-00.nc'

# Dan sam 1 (circle)
dansam1_circle_zipfile = 'dansam1_circle_17Sep2024_12-50-01.zip'
dansam1_circle_file = 'dansam1_circle_17Sep2024_12-50-01.nc'

# Dan sam 2 (square)
dansam2_square_zipfile = 'dansam2_square_17Sep2024_12-50-03.zip'
dansam2_square_file = 'dansam2_square_17Sep2024_12-50-03.nc'

# Dan sam 2 (circle)
dansam2_circle_zipfile = 'dansam2_circle_17Sep2024_12-50-04.zip'
dansam2_circle_file = 'dansam2_circle_17Sep2024_12-50-04.nc'

# =========================
# Core classes & functions
# =========================
class ExperimentManager:
    def __init__(self, ground_truth: xr.Dataset):
        """
        ExperimentManager assumes a circlular wafer and will exclude points outside radius
        """
        self._ground_truth = ground_truth
        self._seensamples = []

        self._radius = self._ground_truth.attrs['shape_width'] / 2
        self._center = self._ground_truth.attrs['shape_center']
        self._resolution = self._ground_truth.attrs['resolution']
        self._q = self._ground_truth.attrs['Q']
        self._coords_valid = self._ground_truth.coords_valid.data
        self._ground_truth_labels = self._ground_truth.ground_truth_labels.data

        # turn string of lists into a list of lists for ground truth legend
        self._phases_present = [ast.literal_eval(i) for i in self._ground_truth.attrs['ground_truth_phasespresent']]
        self._unique_phasecombinations = [ast.literal_eval(i) for i in self._ground_truth.attrs['ground_truth_uniquecombs']]

        # for plotting ground truth
        self._label_names = [ast.literal_eval(i) for i in self._ground_truth.attrs['ground_truth_uniquecombs']]
        self._label_nums = sorted(set(self._ground_truth.ground_truth_labels.data))
        self._phase_label_mapping = dict(zip(map(tuple, self._label_names), self._label_nums))

        # stuff we need for experiment
        self._coords_notmeasured = copy.deepcopy(self._coords_valid)
        self._coords_measured = [] # coords_measured is a list of tuples where measurements have been done

        # dist between coords used for edge detection (better to compute this once up front than in experiment)
        self._dist_btwn_coords = np.linalg.norm(self._coords_valid[:, np.newaxis] - self._coords_valid, axis=2)

    def _get_value_at_coordinates(self, x_coord, y_coord, dataarray):
        """ given x and y, return a numpy array (Iq) """
        interpolated_value = self._ground_truth[dataarray].interp({"x": x_coord, "y": y_coord}).data

        if np.isnan(interpolated_value).any():
            raise ValueError(f'x,y input ({x_coord}, {y_coord}) is outside sample bounds')
        return interpolated_value

    def _get_iq_at_coordinates(self, x_coord, y_coord):
        """ given x and y, return a tuple of numpy arrays (q, Iq) """
        interpolated_value = self._ground_truth['iq'].interp({"x": x_coord, "y": y_coord}).data
        if np.isnan(interpolated_value).any():
            raise ValueError(f'x,y input ({x_coord}, {y_coord}) is outside sample bounds')
        return self._q, interpolated_value

    def get_random_unmeasured_points(self, num_points):
        rng = np.random.default_rng()
        random_indices = rng.choice(self._coords_notmeasured.shape[0], num_points, replace=False)
        random_points = self._coords_notmeasured[random_indices]
        return random_indices, random_points

    def measure(self, xy):
        """ collects data from coordinates (x,y), adds to 'sample' archive """
        x, y = xy
        iq_value = self._get_value_at_coordinates(x, y, dataarray='iq') # this will raise if NaN returned

        if np.any(np.all(self._coords_valid == xy, axis=1)):
            self._coords_measured.append((xy))
            self._coords_notmeasured = self._coords_notmeasured[~np.all(self._coords_notmeasured == xy, axis=1)]
        else:
            print(f"{xy} not a discretized point - data collected but NOT recorded in _coords_measured and NOT removed from _coords_notmeasured")

        new_sample = xr.Dataset(
            {'iq': (('index', 'intensity'), [iq_value])},
            coords={'x': (('index',), [x]), 'y': (('index',), [y])}
        )
        self._seensamples.append(new_sample)

    def _plot_ground_truth(self, colormap = 'Set2', marker='s',marker_size=20, figsize=(10,6), alpha=0.5, legend_on=True, tight_layout=True, title='auto'):
        colors = plt.get_cmap(colormap)
        cmap = mcolors.ListedColormap(colors.colors[:len(set(self._ground_truth.ground_truth_labels.data))])
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(set(self._ground_truth.ground_truth_labels.data)), 1), ncolors=cmap.N)

        if figsize:
            plt.figure(figsize=(figsize[0],figsize[1]))
        else:
            plt.figure()
        scatter = plt.scatter(self._ground_truth.coords_valid.data[:,0], self._ground_truth.coords_valid.data[:,1], c=self._ground_truth.ground_truth_labels.data, cmap=cmap, norm=norm, marker=marker, s=marker_size, alpha=alpha)
        plt.xlabel('x'); plt.ylabel('y')

        if title:
            plt.title('Ground truth' if title == 'auto' else title)

        if legend_on is True:
            legend_elements = []
            for name, num in zip(self._label_names,self._label_nums):
                color = cmap(num)
                legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, label=name))
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1), loc='upper left')

        if tight_layout is True:
            plt.tight_layout()

    def _get_ground_truth_labels(self):
        return self._ground_truth.ground_truth_labels.data

    @property
    def sample(self):
        if len(self._seensamples) == 0:
            return xr.Dataset(
                {'iq': (('index', 'intensity'), np.empty((0, self._ground_truth['iq'].shape[2])))},
                coords={'x': (('index',), []),'y': (('index',), [])},
                attrs={'num_measurements': 0}
            )
        sample = xr.concat(self._seensamples, dim='index')
        sample.attrs['num_measurements'] = len(self._seensamples)
        sample.attrs['Q'] = self._q
        return sample

# ----- Utility functions (unchanged logic) -----
def measurement_dist_calc(measured_coords, notmeasured_coords):
    distances = distance_matrix(notmeasured_coords, measured_coords)
    minimum_distances = np.array([i.min() for i in distances])
    return minimum_distances

def compute_pred_dist(min_dist, measured_coords, notmeasured_coords, initial_guess=(1, 1, 1, 1, 1, 1), print_polyparams=False):
    x = np.array(measured_coords)[:,0]
    y = np.array(measured_coords)[:,1]
    meas_coords = np.column_stack((x,y))
    def poly_func(meas_coords, a, b, c, d, e, f):
        return a*meas_coords[:,0]**2 + b*meas_coords[:,1]**2 + c*meas_coords[:,0]*meas_coords[:,1] + d*meas_coords[:,0] + e*meas_coords[:,1] + f
    params, _ = curve_fit(poly_func, meas_coords, min_dist, p0=initial_guess)
    if print_polyparams: print(f"polynomial parameters: {params}")
    predicted_distances = poly_func(notmeasured_coords, *params)
    return predicted_distances

def dist_transform_normalize(distances, sigma=2, dist_power_scale=1, normalize=True):
    if np.isnan(distances).any():
        raise ValueError(f'NaN found in distances - distances must be an array of valid points (e.g., within bounds)')
    dist = 1-np.exp(-distances**2/ (2*sigma**2))
    dist[dist<=0] = 0
    dist = dist**dist_power_scale
    if normalize:
        dist_sum = dist.sum()
        if dist_sum != 0:
            dist /= dist_sum
        else:
            dist = np.ones_like(dist) / len(dist)
    return dist

def get_cdf(dist):
    return np.cumsum(dist)

def sample_from_numberline(cdf, num_suggested_points, side='right'):
    r = np.random.rand(num_suggested_points)
    idx = np.searchsorted(cdf, r, side=side)
    return idx

def get_next_coords(notmeasured_coords, idx):
    next_coords = np.array([notmeasured_coords[i] for i in idx])
    return next_coords

def suggest_next_coords(distances, notmeasured_coords, sigma=2, dist_power_scale=1, num_suggested_points=1,
                        display_distances=False, display_dist=False, display_cdf=False, display_next_coords=False,
                        disp_marker='s', disp_markersize=3, colormap='viridis',disp_figsize=(8,5)):
    dist = dist_transform_normalize(distances, sigma=sigma, dist_power_scale=dist_power_scale)
    cdf = get_cdf(dist)
    idx = sample_from_numberline(cdf, num_suggested_points=num_suggested_points)
    next_coords = get_next_coords(notmeasured_coords, idx=idx)

    if display_distances:
        plt.figure(figsize=disp_figsize if disp_figsize else None)
        plt.scatter(notmeasured_coords[:,0], notmeasured_coords[:,1], c=distances, cmap=colormap, marker=disp_marker, s=disp_markersize)
        plt.colorbar(label='input distances'); plt.title('Input distances')

    if display_dist:
        plt.figure(figsize=disp_figsize if disp_figsize else None)
        plt.scatter(notmeasured_coords[:,0], notmeasured_coords[:,1], c=dist, cmap=colormap, marker=disp_marker, s=disp_markersize)
        plt.colorbar(label='Calculated distances (normalized)'); plt.title('Calculated distances (normalized)')

    if display_cdf:
        plt.figure(figsize=disp_figsize if disp_figsize else None)
        plt.plot(cdf); plt.title('CDF')

    if display_next_coords:
        plt.figure(figsize=disp_figsize if disp_figsize else None)
        plt.scatter(notmeasured_coords[:,0], notmeasured_coords[:,1], c=dist, cmap=colormap, marker=disp_marker, s=disp_markersize)
        plt.colorbar(label='Calculated distances (normalized)')
        plt.scatter(next_coords[:,0], next_coords[:,1], marker='x', color='red')
        plt.title('Suggested measurement(s)')

    return next_coords

def match_labels(ground_truth_labels, predicted_labels):
    max_label = max(ground_truth_labels.max(), predicted_labels.max()) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)
    for true, pred in zip(ground_truth_labels, predicted_labels):
        confusion_matrix[true, pred] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    label_mapping = dict(zip(col_ind, row_ind))
    predicted_labels_mapped = np.array([label_mapping[p] for p in predicted_labels])
    return predicted_labels_mapped

def get_pred_accuracy(ground_truth_labels, aligned_predicted_labels, print_accuracy_stats=False):
    if print_accuracy_stats:
        count_match = int(np.sum(aligned_predicted_labels == ground_truth_labels))
        count_notmatch = len(ground_truth_labels) - count_match
        print(f"label matches: {count_match} / {len(ground_truth_labels)}\nnum not matching: {count_notmatch} / {len(ground_truth_labels)}\nAccuracy: {count_match / len(ground_truth_labels) * 100}")
    accuracy = np.mean(aligned_predicted_labels == ground_truth_labels)
    return accuracy

def get_mislabeled_coords(ground_truth_labels, aligned_predicted_labels, valid_coords):
    mislabeled_indices = np.where(ground_truth_labels != aligned_predicted_labels)[0]
    mislabeled_coords = valid_coords[mislabeled_indices]
    return mislabeled_coords

def geometric_spacing(xmin, xmax, ymin, ymax, n, seen_list = [], cen_xy = (0,0), rmax=-1):
    nuse = int(2**n+1)
    xlist = np.linspace(xmin, xmax, nuse)
    ylist = np.linspace(ymin, ymax, nuse)
    my_list = []
    def dist_good(xy, return_val = False):
        if rmax < 0: return True
        dx = abs(cen_xy[0] - xy[0]); dy = abs(cen_xy[1] - xy[1])
        if return_val: return np.sqrt(dx**2. + dy**2.)
        return (np.sqrt(dx**2. + dy**2.) <= rmax)
    check_seen_list = []
    for xy in np.array(seen_list).T:
        check_seen_list.append(tuple(xy))
    for y in ylist:
        for x in xlist:
            coords = [np.round(x,2),np.round(y,2)]
            if dist_good(coords):
                if tuple(coords) not in check_seen_list:
                    my_list.append(coords)
    return np.array(my_list).T

def geoseries(xmin, xmax, ymin, ymax, nmax, cen_xy = (0,0), rmax=-1):
    this = geometric_spacing(xmin,xmax,ymin,ymax,0, cen_xy = cen_xy, rmax=rmax)
    for nis in range(nmax):
        this2 = geometric_spacing(xmin,xmax,ymin,ymax,nis+1, seen_list=this, cen_xy = cen_xy, rmax=rmax)
        if len(this) >0:
            this = np.concatenate([this.T, this2.T]).T
        else:
            this = this2
    return this

def geoseries_to_N(xmin, xmax, ymin, ymax, N, cen_xy = (0,0), rmax=-1):
    go_on = False
    use_n = 0
    while not go_on:
        this_series = geoseries(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, nmax=use_n, cen_xy = cen_xy, rmax=rmax)
        use_n += 1
        if len(this_series.T) >= N:
            go_on = True
    return this_series.T[:N].T

def detect_edges(coords, labels, threshold):
    edge_detect = np.zeros(len(coords), dtype=bool)
    edge_ratio = np.zeros(len(coords), dtype=float)
    for i, coord in enumerate(coords):
        distances = np.linalg.norm(coords - coord, axis=1)
        neighbor_mask = (distances > 0) & (distances <= threshold)
        neighbor_indices = np.where(neighbor_mask)[0]
        if len(neighbor_indices) > 0:
            neighbor_labels = labels[neighbor_indices]
            current_label = labels[i]
            different_neighbors = neighbor_labels != current_label
            edge_detect[i] = np.any(different_neighbors)
            edge_ratio[i] = np.sum(different_neighbors) / len(neighbor_indices)
    return edge_detect, edge_ratio

def detect_edges_with_distances(distances, labels, threshold):
    edge_detect = np.zeros(len(labels), dtype=bool)
    edge_ratio = np.zeros(len(labels), dtype=float)
    for i in range(len(labels)):
        neighbor_mask = (distances[i] > 0) & (distances[i] <= threshold)
        neighbor_indices = np.where(neighbor_mask)[0]
        if len(neighbor_indices) > 0:
            neighbor_labels = labels[neighbor_indices]
            current_label = labels[i]
            different_neighbors = neighbor_labels != current_label
            edge_detect[i] = np.any(different_neighbors)
            edge_ratio[i] = np.sum(different_neighbors) / len(neighbor_indices)
    return edge_detect, edge_ratio

def sample_and_remove(array):
    if len(array) == 0:
        raise ValueError("Array is empty. No more elements to sample.")
    random_index = np.random.randint(len(array))
    sampled_coordinate = array[random_index]
    array = np.delete(array, random_index, axis=0)
    return sampled_coordinate, array

def prob_norm(array):
    array_sum = array.sum()
    if array_sum != 0:
        array_norm = array / array_sum
    else:
        array_norm = np.ones_like(array) / len(array)
    return array_norm

def minmax_norm(array):
    array_norm = (array - array.min()) / (array.max() - array.min())
    return array_norm

# =========================
# Main execution
# =========================
def main():
    _runtime_sanity() # Check threadpools and affinities

    parser = argparse.ArgumentParser(description="Run simulated experiment(s) on CPU/Slurm")
    parser.add_argument("--dataset", default="complex",
                        choices=["complex","simpleLR","dansam1_square","dansam1_circle","dansam2_square","dansam2_circle"],
                        help="Which dataset to use")
    parser.add_argument("--campaign", type=str, default="default",
                    help="Short tag to group outputs across reruns")
    parser.add_argument("--total-meas", type=int, default=805, help="Total measurements to do")
    parser.add_argument("--run-id", type=int, default=None, help="Single run id (e.g., from $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--saved-data-pkl", type=str, default="light", help="Bookkeeping to save")
    parser.add_argument("--approach", type=str, default="geoseries", help="Acquisition approach to use")
    args = parser.parse_args()

    # If running inside a Slurm array, adopt that index
    if args.run_id is None:
        slurm_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        num_exp_runs = [int(slurm_idx)] if slurm_idx is not None else [0]
    else:
        num_exp_runs = [args.run_id]

    script_start_time = time.time()
    exp_loop_start_time = time.time()

    for exp_run_num in num_exp_runs:

        total_meas_todo = int(args.total_meas)
        exp_run = f'seed49_mmm1_{exp_run_num}'
        approach = args.approach
        saved_data_pkl = args.saved_data_pkl

        print(f'\n\n\n###############\nStarting {approach} experiment {exp_run_num}!\n###############\n\n\n', flush=True)

        # Dataset selection
        if args.dataset == 'complex':
            zipfile, file = complex_zipfile, complex_file
        elif args.dataset == 'simpleLR':
            zipfile, file = simpleLR_zipfile, simpleLR_file
        elif args.dataset == 'dansam1_square':
            zipfile, file = dansam1_square_zipfile, dansam1_square_file
        elif args.dataset == 'dansam1_circle':
            zipfile, file = dansam1_circle_zipfile, dansam1_circle_file
        elif args.dataset == 'dansam2_square':
            zipfile, file = dansam2_square_zipfile, dansam2_square_file
        elif args.dataset == 'dansam2_circle':
            zipfile, file = dansam2_circle_zipfile, dansam2_circle_file
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        # Ensure dataset is extracted (simple, non-locking)
        if file not in os.listdir(path):
            with ZipFile(os.path.join(path,zipfile), 'r') as zObject:
                zObject.extract(file, path=path)
        ground_truth = xr.open_dataset(os.path.join(path,file)) # read in dataset

        experiment = ExperimentManager(ground_truth) # initialize experiment

        # Instantiate kmeans
        kmeans = KMeans(n_clusters=3, random_state=42)

        # Define the gaussian process kernel
        kernel = C(1.0, (1e-3, 10)) * RBF(1.0, (1e-3, 30))
        # kernel = C(1.0, (1e-3, 10)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3,30), nu=0.5)

        # Instantiate the classifier (threads = 1 so that slurm handles the parallelism)
        gpc = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=1, n_restarts_optimizer=5)

        # get ground_truth labels
        ground_truth_labels = experiment._get_ground_truth_labels()

        # model labels (from GP prediction after training on all data)
        try:
            gp_df = pd.read_csv(os.path.join(path, 'gp_prediction_onalldata.csv'))
            ground_truth_labels_gp = gp_df.predicted_labels.values
        except Exception:
            raise ValueError("Error loading gp_prediction_onalldata.csv - ensure the file exists and is formatted correctly")

        # Geometric series setup
        xmin = experiment._coords_valid[:,0].min()
        xmax = experiment._coords_valid[:,0].max()
        ymin = experiment._coords_valid[:,1].min()
        ymax = experiment._coords_valid[:,1].max()
        cen_xy = experiment._ground_truth.attrs['shape_center']
        rmax = ymax - 0.4 # 1 pixel in from the edge for this series
        geox, geoy = geoseries(xmin = xmin, xmax = xmax,
                        ymin = ymin, ymax = ymax,
                        nmax = 6, # (5=797, 6=3209, 7=12853)
                        cen_xy=cen_xy,rmax=ymax)
        geotomeas = np.column_stack((geox,geoy)) # list to measure from, in order

        # series of points
        geotomeas_1 = geotomeas[0:13]
        geotomeas_2 = geotomeas[13:49]
        geotomeas_3 = geotomeas[49:197]
        geotomeas_4 = geotomeas[197:797]
        geotomeas_5 = geotomeas[797:3209]

        ###################################
        # Bookkeeping
        ###################################
        measured_labels = []
        num_measurements = []

        # 2 diff accuracies
        accuracies = [] # measured (N) + model (Ntot - N) / measured (Ntot)
        accuracies_gpgt = [] # model(N) + model (Ntot - N) / Model (Ntot)

        # prediction metrics (from predicting all)
        f1s_weighted_predall = []
        f1s_macro_predall = []
        precisions_weighted_predall = []
        precisions_macro_predall = []
        recalls_weighted_predall = []
        recalls_macro_predall = []

        # gp timing and kernel prms
        gp_fittime = []
        gp_kernelprms = []
        gp_predtime_unmeas = []
        gp_probtime_unmeas = []
        gp_predtime_all = []
        gp_probtime_all = []

        # prediction on unmeas (gt labels)
        gp_predictions_unmeas = []
        gp_probabilities_unmeas = []
        gp_confidences_unmeas = []
        gp_uncertainties_unmeas = []

        # prediction on all (gt labels)
        gp_predictions_all = []
        gp_probabilities_all = []
        gp_confidences_all = []
        gp_uncertainties_all = []

        # labels
        all_combined_labels = []

        coords_measured = []
        coords_notmeasured = []
        coords_mislabeled_predunmeas = []
        coords_mislabeled_predall = []

        # bookkeeping of measurement suggestions
        gp_unc_01s = []
        gp_unc_norms = []
        gp_unc_cdfs = []

        # edge weighted
        unmeas_edges_ints = []
        ew_unc_01s = []
        ew_unc_norms = []
        ew_unc_cdfs = []

        # edge ratio weighted
        unmeas_edges_ratios = []
        rw_unc_01s = []
        rw_unc_norms = []
        rw_unc_cdfs = []

        # meas_distance (not used, but may be informative)
        meas_distances = []
        meas_dist_norms = []
        meas_dist_cdfs = []

        # separate bookkeeping for cdf, idx, coords for measurement suggestion
        all_cdfs = []
        all_idxs = []
        all_next_coords = []

        # initial measurements (seed with 49 pts)
        init_meas = np.vstack((geotomeas_1,geotomeas_2))
        for i in range(init_meas.shape[0]):
            geodist = np.linalg.norm(experiment._coords_notmeasured - init_meas[i], axis=1)
            closest_idx = np.argmin(geodist)
            experiment.measure((experiment._coords_notmeasured[:,0][closest_idx], experiment._coords_notmeasured[:,1][closest_idx]))
        print(f"\nInitial {init_meas.shape[0]} measurements done\n", flush=True)

        # Measurement loop
        for i in range(experiment.sample.attrs['num_measurements'], total_meas_todo):
            print(f'\n{approach} exp {exp_run_num}: starting measurement {i + 1} / {total_meas_todo}...\nTotal elapsed time: {(time.time() - script_start_time) / 60} min.\n', flush=True)

            #############################
            # bookkeeping measurements
            #############################
            coords_measured.append(np.array(experiment._coords_measured))
            coords_notmeasured.append(experiment._coords_notmeasured)
            num_measurements.append(experiment.sample.attrs['num_measurements'])

            ###########################################
            # get labels from ground truth directly
            ###########################################
            measured_indices = []
            for xy in np.array(experiment._coords_measured):
                match = np.where((experiment._coords_valid == xy).all(axis=1))[0]
                if match.size > 0:
                    measured_indices.append(match[0])
                else:
                    raise ValueError(f'measured point {xy} not found in valid coords - inspect for issue')
            measured_indices = np.array(measured_indices)
            labels = ground_truth_labels[measured_indices]
            measured_labels.append(labels)

            ################################################
            # GP fitting
            ################################################
            start_time = time.time()
            gpc.fit(np.array(experiment._coords_measured), labels)
            gp_fittime.append((time.time() - start_time))
            gp_kernelprms.append(gpc.kernel_.get_params()['kernels'])

            ################################################
            # GP prediction - unmeasured points
            ################################################
            start_time = time.time()
            predicted_labels = gpc.predict(experiment._coords_notmeasured)
            gp_predtime_unmeas.append((time.time() - start_time))

            start_time = time.time()
            probabilities = gpc.predict_proba(experiment._coords_notmeasured)
            gp_probtime_unmeas.append((time.time() - start_time))
            confidence = np.max(probabilities, axis=1)
            uncertainty = entropy(probabilities.T)

            gp_predictions_unmeas.append(predicted_labels)
            gp_probabilities_unmeas.append(probabilities)
            gp_confidences_unmeas.append(confidence)
            gp_uncertainties_unmeas.append(uncertainty)

            # combine measured GT labels + predicted labels for unmeasured
            measured_tuples = {tuple(coord): i for i, coord in enumerate(np.array(experiment._coords_measured))}
            notmeasured_tuples = {tuple(coord): i for i, coord in enumerate(experiment._coords_notmeasured)}

            combined_labels = []
            for coord in experiment._coords_valid:
                coord_tuple = tuple(coord)
                if coord_tuple in measured_tuples:
                    idx = measured_tuples[coord_tuple]
                    combined_labels.append(labels[idx])
                elif coord_tuple in notmeasured_tuples:
                    idx = notmeasured_tuples[coord_tuple]
                    combined_labels.append(predicted_labels[idx])
                else:
                    raise ValueError(f'coord {coord} not found in coords_measured or coords_notmeasured')
            combined_labels = np.array(combined_labels)
            all_combined_labels.append(combined_labels)
            mislabeled_coords = get_mislabeled_coords(ground_truth_labels, combined_labels, experiment._coords_valid)
            coords_mislabeled_predunmeas.append(mislabeled_coords)
            accuracy = get_pred_accuracy(ground_truth_labels, combined_labels)
            accuracies.append(accuracy)

            ################################################
            # GP prediction - all points
            ################################################
            start_time = time.time()
            predicted_labels = gpc.predict(experiment._coords_valid)
            gp_predtime_all.append((time.time() - start_time))

            start_time = time.time()
            probabilities = gpc.predict_proba(experiment._coords_valid)
            gp_probtime_all.append((time.time() - start_time))
            confidence = np.max(probabilities, axis=1)
            uncertainty = entropy(probabilities.T)

            gp_predictions_all.append(predicted_labels)
            gp_probabilities_all.append(probabilities)
            gp_confidences_all.append(confidence)
            gp_uncertainties_all.append(uncertainty)

            mislabeled_coords = get_mislabeled_coords(ground_truth_labels_gp, predicted_labels, experiment._coords_valid)
            coords_mislabeled_predall.append(mislabeled_coords)

            accuracy = get_pred_accuracy(ground_truth_labels_gp, predicted_labels)
            f1_w = f1_score(ground_truth_labels_gp, predicted_labels, average='weighted')
            precision_w = precision_score(ground_truth_labels_gp, predicted_labels, average='weighted')
            recall_w = recall_score(ground_truth_labels_gp, predicted_labels, average='weighted')
            f1_m = f1_score(ground_truth_labels_gp, predicted_labels, average='macro')
            precision_m = precision_score(ground_truth_labels_gp, predicted_labels, average='macro')
            recall_m = recall_score(ground_truth_labels_gp, predicted_labels, average='macro')

            accuracies_gpgt.append(accuracy)
            f1s_weighted_predall.append(f1_w)
            f1s_macro_predall.append(f1_m)
            precisions_weighted_predall.append(precision_w)
            precisions_macro_predall.append(precision_m)
            recalls_weighted_predall.append(recall_w)
            recalls_macro_predall.append(recall_m)

            ################################################
            # get suggested measurement - use unmeasured
            ################################################
            edge_detect, edge_ratio = detect_edges_with_distances(experiment._dist_btwn_coords,
                                            combined_labels,
                                            threshold=experiment._ground_truth.attrs['resolution'] * 1.5)
            all_indices = np.arange(experiment._coords_valid.shape[0])
            unmeasured_indices = np.setdiff1d(all_indices, measured_indices)
            unmeasured_labels = combined_labels[unmeasured_indices]

            unmeas_edges = edge_detect[unmeasured_indices]
            unmeas_edges_int = unmeas_edges.astype(int)
            unmeas_edges_ratio = edge_ratio[unmeasured_indices]

            gp_unc = gp_uncertainties_unmeas[-1]
            gp_unc_01 = minmax_norm(gp_unc)
            gp_unc_norm = prob_norm(gp_unc_01)
            gp_unc_cdf = get_cdf(gp_unc_norm)

            # edge weighted uncertainty
            ew_unc_01 = minmax_norm(minmax_norm(gp_unc) + minmax_norm(unmeas_edges_int))
            ew_unc_norm = prob_norm(ew_unc_01)
            ew_unc_cdf = get_cdf(ew_unc_norm)

            # edge ratio weighted uncertainty
            rw_unc_01 = minmax_norm(minmax_norm(gp_unc) + minmax_norm(unmeas_edges_ratio))
            rw_unc_norm = prob_norm(rw_unc_01)
            rw_unc_cdf = get_cdf(rw_unc_norm)

            # proximity / meas distances
            meas_distance = measurement_dist_calc(measured_coords=experiment._coords_measured, notmeasured_coords=experiment._coords_notmeasured)
            meas_dist_norm = dist_transform_normalize(meas_distance, sigma=2, dist_power_scale=2, normalize=True)
            meas_cdf = get_cdf(meas_dist_norm)

            #################################
            # Measurement selection strategy
            #################################

            if approach == 'geoseries':
                # Random sampling of geometric series
                if i in np.arange(0,13):
                    coord_to_meas, geotomeas_1 = sample_and_remove(geotomeas_1)
                elif i in np.arange(13,49):
                    coord_to_meas, geotomeas_2 = sample_and_remove(geotomeas_2)
                elif i in np.arange(49,197):
                    coord_to_meas, geotomeas_3 = sample_and_remove(geotomeas_3)
                elif i in np.arange(197,797):
                    coord_to_meas, geotomeas_4 = sample_and_remove(geotomeas_4)
                elif i in np.arange(797,3209):
                    coord_to_meas, geotomeas_5 = sample_and_remove(geotomeas_5)
                else:
                    raise ValueError(f'Geoseries approach exhausted - no more points to measure - total measurements requested: {total_meas_todo}, current measurement: {i+1}')

                geodist = np.linalg.norm(experiment._coords_notmeasured - coord_to_meas, axis=1)
                idx = np.argmin(geodist)
                next_coords = np.array([[experiment._coords_notmeasured[idx,0],experiment._coords_notmeasured[idx,1]]])
                #next_coords = (experiment._coords_notmeasured[:,0][idx], experiment._coords_notmeasured[:,1][idx])
                cdf = ew_unc_cdf # placeholder for bookkeeping

            elif approach == 'random':
                idx, next_coords = experiment.get_random_unmeasured_points(1) # now returns coords and idx
                cdf = ew_unc_cdf # placeholder for bookkeeping

            elif approach == 'gpewunc':
                # choose next coords from edge-weighted cdf
                cdf = ew_unc_cdf
                idx = sample_from_numberline(cdf, num_suggested_points=1, side='right')
                next_coords = get_next_coords(notmeasured_coords=experiment._coords_notmeasured, idx=idx)

            elif approach == 'gprwunc':
                # choose next coords from ratio-weighted cdf
                cdf = rw_unc_cdf
                idx = sample_from_numberline(cdf, num_suggested_points=1, side='right')
                next_coords = get_next_coords(notmeasured_coords=experiment._coords_notmeasured, idx=idx)

            else:
                raise ValueError(f"Unknown approach: {approach}")

            # Bookkeeping of measurement suggestions
            all_cdfs.append(cdf)
            all_idxs.append(idx)
            all_next_coords.append(next_coords)

            gp_unc_01s.append(gp_unc_01); gp_unc_norms.append(gp_unc_norm); gp_unc_cdfs.append(gp_unc_cdf)
            unmeas_edges_ints.append(unmeas_edges_int); ew_unc_01s.append(ew_unc_01); ew_unc_norms.append(ew_unc_norm); ew_unc_cdfs.append(ew_unc_cdf)
            unmeas_edges_ratios.append(unmeas_edges_ratio); rw_unc_01s.append(rw_unc_01); rw_unc_norms.append(rw_unc_norm); rw_unc_cdfs.append(rw_unc_cdf)
            meas_distances.append(meas_distance); meas_dist_norms.append(meas_dist_norm); meas_dist_cdfs.append(meas_cdf)

            # do measurement
            for x,y in next_coords:
                experiment.measure((x,y))

        print(f"\n{total_meas_todo} measurements done - starting end of run bookkeeping\n", flush=True)

        # End of run bookkeeping (mirrors loop body ending)
        coords_measured.append(np.array(experiment._coords_measured))
        coords_notmeasured.append(experiment._coords_notmeasured)
        num_measurements.append(experiment.sample.attrs['num_measurements'])

        measured_indices = []
        for xy in np.array(experiment._coords_measured):
            match = np.where((experiment._coords_valid == xy).all(axis=1))[0]
            if match.size > 0:
                measured_indices.append(match[0])
            else:
                raise ValueError(f'measured point {xy} not found in valid coords - inspect for issue')
        measured_indices = np.array(measured_indices)
        labels = ground_truth_labels[measured_indices]
        measured_labels.append(labels)

        start_time = time.time()
        gpc.fit(np.array(experiment._coords_measured), labels)
        gp_fittime.append((time.time() - start_time))
        gp_kernelprms.append(gpc.kernel_.get_params()['kernels'])

        start_time = time.time()
        predicted_labels = gpc.predict(experiment._coords_notmeasured)
        gp_predtime_unmeas.append((time.time() - start_time))

        start_time = time.time()
        probabilities = gpc.predict_proba(experiment._coords_notmeasured)
        gp_probtime_unmeas.append((time.time() - start_time))
        confidence = np.max(probabilities, axis=1)
        uncertainty = entropy(probabilities.T)

        gp_predictions_unmeas.append(predicted_labels)
        gp_probabilities_unmeas.append(probabilities)
        gp_confidences_unmeas.append(confidence)
        gp_uncertainties_unmeas.append(uncertainty)

        measured_tuples = {tuple(coord): i for i, coord in enumerate(np.array(experiment._coords_measured))}
        notmeasured_tuples = {tuple(coord): i for i, coord in enumerate(experiment._coords_notmeasured)}
        combined_labels = []
        for coord in experiment._coords_valid:
            coord_tuple = tuple(coord)
            if coord_tuple in measured_tuples:
                idx = measured_tuples[coord_tuple]
                combined_labels.append(labels[idx])
            elif coord_tuple in notmeasured_tuples:
                idx = notmeasured_tuples[coord_tuple]
                combined_labels.append(predicted_labels[idx])
            else:
                raise ValueError(f'coord {coord} not found in coords_measured or coords_notmeasured')
        combined_labels = np.array(combined_labels)
        all_combined_labels.append(combined_labels)
        mislabeled_coords = get_mislabeled_coords(ground_truth_labels, combined_labels, experiment._coords_valid)
        coords_mislabeled_predunmeas.append(mislabeled_coords)
        accuracy = get_pred_accuracy(ground_truth_labels, combined_labels)
        accuracies.append(accuracy)

        start_time = time.time()
        predicted_labels = gpc.predict(experiment._coords_valid)
        gp_predtime_all.append((time.time() - start_time))

        start_time = time.time()
        probabilities = gpc.predict_proba(experiment._coords_valid)
        gp_probtime_all.append((time.time() - start_time))
        confidence = np.max(probabilities, axis=1)
        uncertainty = entropy(probabilities.T)

        gp_predictions_all.append(predicted_labels)
        gp_probabilities_all.append(probabilities)
        gp_confidences_all.append(confidence)
        gp_uncertainties_all.append(uncertainty)

        mislabeled_coords = get_mislabeled_coords(ground_truth_labels_gp, predicted_labels, experiment._coords_valid)
        coords_mislabeled_predall.append(mislabeled_coords)

        accuracy = get_pred_accuracy(ground_truth_labels_gp, predicted_labels)
        f1_w = f1_score(ground_truth_labels_gp, predicted_labels, average='weighted')
        precision_w = precision_score(ground_truth_labels_gp, predicted_labels, average='weighted')
        recall_w = recall_score(ground_truth_labels_gp, predicted_labels, average='weighted')
        f1_m = f1_score(ground_truth_labels_gp, predicted_labels, average='macro')
        precision_m = precision_score(ground_truth_labels_gp, predicted_labels, average='macro')
        recall_m = recall_score(ground_truth_labels_gp, predicted_labels, average='macro')
        accuracies_gpgt.append(accuracy)
        f1s_weighted_predall.append(f1_w)
        f1s_macro_predall.append(f1_m)
        precisions_weighted_predall.append(precision_w)
        precisions_macro_predall.append(precision_m)
        recalls_weighted_predall.append(recall_w)
        recalls_macro_predall.append(recall_m)

        edge_detect, edge_ratio = detect_edges_with_distances(experiment._dist_btwn_coords,
                                        combined_labels,
                                        threshold=experiment._ground_truth.attrs['resolution'] * 1.5)
        all_indices = np.arange(experiment._coords_valid.shape[0])
        unmeasured_indices = np.setdiff1d(all_indices, measured_indices)
        unmeasured_labels = combined_labels[unmeasured_indices]

        unmeas_edges = edge_detect[unmeasured_indices]
        unmeas_edges_int = unmeas_edges.astype(int)
        unmeas_edges_ratio = edge_ratio[unmeasured_indices]

        gp_unc = gp_uncertainties_unmeas[-1]
        gp_unc_01 = minmax_norm(gp_unc)
        gp_unc_norm = prob_norm(gp_unc_01)
        gp_unc_cdf = get_cdf(gp_unc_norm)

        ew_unc_01 = minmax_norm(minmax_norm(gp_unc) + minmax_norm(unmeas_edges_int))
        ew_unc_norm = prob_norm(ew_unc_01)
        ew_unc_cdf = get_cdf(ew_unc_norm)

        rw_unc_01 = minmax_norm(minmax_norm(gp_unc) + minmax_norm(unmeas_edges_ratio))
        rw_unc_norm = prob_norm(rw_unc_01)
        rw_unc_cdf = get_cdf(rw_unc_norm)

        meas_distance = measurement_dist_calc(measured_coords=experiment._coords_measured, notmeasured_coords=experiment._coords_notmeasured)
        meas_dist_norm = dist_transform_normalize(meas_distance, sigma=2, dist_power_scale=2, normalize=True)
        meas_cdf = get_cdf(meas_dist_norm)

        #################################
        # Measurement selection strategy - just for getting last cdf, idx, coords
        #################################

        if approach == 'geoseries':
            # Random sampling of geometric series
            if i+1 in np.arange(0,13):
                coord_to_meas, geotomeas_1 = sample_and_remove(geotomeas_1)
            elif i+1 in np.arange(13,49):
                coord_to_meas, geotomeas_2 = sample_and_remove(geotomeas_2)
            elif i+1 in np.arange(49,197):
                coord_to_meas, geotomeas_3 = sample_and_remove(geotomeas_3)
            elif i+1 in np.arange(197,797):
                coord_to_meas, geotomeas_4 = sample_and_remove(geotomeas_4)
            elif i+1 in np.arange(797,3209):
                coord_to_meas, geotomeas_5 = sample_and_remove(geotomeas_5)
            else:
                print(f'Geoseries approach exhausted - no more points to measure - total measurements requested: {total_meas_todo}, current measurement: {i+1}')
                print('Setting coord_to_meas to (0,0) for bookkeeping purposes')
                coord_to_meas = (0,0) # dummy value for bookkeeping

            geodist = np.linalg.norm(experiment._coords_notmeasured - coord_to_meas, axis=1)
            idx = np.argmin(geodist)
            next_coords = np.array([[experiment._coords_notmeasured[idx,0],experiment._coords_notmeasured[idx,1]]])
            #next_coords = (experiment._coords_notmeasured[:,0][idx], experiment._coords_notmeasured[:,1][idx])
            cdf = ew_unc_cdf # placeholder for bookkeeping

        elif approach == 'random':
            idx, next_coords = experiment.get_random_unmeasured_points(1) # now returns coords and idx
            cdf = ew_unc_cdf # placeholder for bookkeeping

        elif approach == 'gpewunc':
            # choose next coords from edge-weighted cdf
            cdf = ew_unc_cdf
            idx = sample_from_numberline(cdf, num_suggested_points=1, side='right')
            next_coords = get_next_coords(notmeasured_coords=experiment._coords_notmeasured, idx=idx)

        elif approach == 'gprwunc':
            # choose next coords from ratio-weighted cdf
            cdf = rw_unc_cdf
            idx = sample_from_numberline(cdf, num_suggested_points=1, side='right')
            next_coords = get_next_coords(notmeasured_coords=experiment._coords_notmeasured, idx=idx)

        else:
            raise ValueError(f"Unknown approach: {approach}")

        # Bookkeeping of measurement suggestions
        all_cdfs.append(cdf)
        all_idxs.append(idx)
        all_next_coords.append(next_coords)

        gp_unc_01s.append(gp_unc_01); gp_unc_norms.append(gp_unc_norm); gp_unc_cdfs.append(gp_unc_cdf)
        unmeas_edges_ints.append(unmeas_edges_int); ew_unc_01s.append(ew_unc_01); ew_unc_norms.append(ew_unc_norm); ew_unc_cdfs.append(ew_unc_cdf)
        unmeas_edges_ratios.append(unmeas_edges_ratio); rw_unc_01s.append(rw_unc_01); rw_unc_norms.append(rw_unc_norm); rw_unc_cdfs.append(rw_unc_cdf)
        meas_distances.append(meas_distance); meas_dist_norms.append(meas_dist_norm); meas_dist_cdfs.append(meas_cdf)

        print(f"\nbookkeeping done - setting up dict for pkling data\n", flush=True)

        ####################################
        # Saving results
        ####################################
        if 'dansam1' in zipfile or 'dansam2' in zipfile:
            words = zipfile.split('_',-1)
            fname = '-'.join(words[:2])
        elif 'complex' in zipfile:
            fname = 'complex'
        elif 'simpleLR' in zipfile:
            fname = 'simpleLR'
        else:
            fname = Path(file).stem

        if saved_data_pkl == 'full':
            data = {
                'ground_truth_labels': ground_truth_labels,
                'ground_truth_labels_gp': ground_truth_labels_gp,
                'num_measurements': num_measurements,
                'percent_measured': (np.array(num_measurements) / ground_truth_labels.shape[0]) * 100,
                'accuracies': accuracies,
                'accuracies_gpgt': accuracies_gpgt,

                # labels
                'measured_labels': measured_labels,
                'all_combined_labels': all_combined_labels,
                'all_labels_predall': gp_predictions_all,

                # coords
                'coords_valid': experiment._coords_valid,
                'coords_measured': coords_measured,
                'coords_notmeasured': coords_notmeasured,

                # mislabeling
                'coords_mislabeled_predunmeas': coords_mislabeled_predunmeas,
                'coords_mislabeled_predall': coords_mislabeled_predall,

                # gp pred unmeas & all
                'gp_predictions_unmeas': gp_predictions_unmeas,
                'gp_probabilities_unmeas': gp_probabilities_unmeas,
                'gp_confidences_unmeas': gp_confidences_unmeas,
                'gp_uncertainties_unmeas': gp_uncertainties_unmeas,
                'gp_predictions_all': gp_predictions_all,
                'gp_probabilities_all': gp_probabilities_all,
                'gp_confidences_all': gp_confidences_all,
                'gp_uncertainties_all': gp_uncertainties_all,

                # gp uncertainty normalized, cdfs
                'gp_unc_01s': gp_unc_01s,
                'gp_unc_norms': gp_unc_norms,
                'gp_unc_cdfs': gp_unc_cdfs,

                # edge weighted normalized, cdfs
                'unmeas_edges_ints': unmeas_edges_ints,
                'ew_unc_01s': ew_unc_01s,
                'ew_unc_norms': ew_unc_norms,
                'ew_unc_cdfs': ew_unc_cdfs,

                # edge ratio weighted normalized, cdfs
                'unmeas_edges_ratios': unmeas_edges_ratios,
                'rw_unc_01s': rw_unc_01s,
                'rw_unc_norms': rw_unc_norms,
                'rw_unc_cdfs': rw_unc_cdfs,

                # meas_distance
                'meas_distances': meas_distances,
                'meas_dist_norms': meas_dist_norms,
                'meas_dist_cdfs': meas_dist_cdfs,

                # cdf/idx/next_coords tracking
                'all_cdfs': all_cdfs,
                'all_idxs': all_idxs,
                'all_next_coords': all_next_coords,

                # gp times and kernel prms
                'gp_fittime': gp_fittime,
                'gp_kernelprms': gp_kernelprms,
                'gp_predtime_unmeas': gp_predtime_unmeas,
                'gp_probtime_unmeas': gp_probtime_unmeas,
                'gp_predtime_all': gp_predtime_all,
                'gp_probtime_all': gp_probtime_all,

                # prediction scores
                'f1s_weighted_predall': f1s_weighted_predall,
                'f1s_macro_predall': f1s_macro_predall,
                'precisions_weighted_predall': precisions_weighted_predall,
                'precisions_macro_predall': precisions_macro_predall,
                'recalls_weighted_predall': recalls_weighted_predall,
                'recalls_macro_predall': recalls_macro_predall
            }

        elif saved_data_pkl == 'light':
            data = {
            'ground_truth_labels': ground_truth_labels,
            'ground_truth_labels_gp': ground_truth_labels_gp,
            'num_measurements': num_measurements,
            'percent_measured': (np.array(num_measurements) / ground_truth_labels.shape[0]) * 100,
            'accuracies': accuracies,
            'accuracies_gpgt': accuracies_gpgt,

            # labels
            'measured_labels': measured_labels,
            'all_combined_labels': all_combined_labels,
            'all_labels_predall': gp_predictions_all,

            # coords
            'coords_valid': experiment._coords_valid,
            'coords_measured': coords_measured,
            'coords_notmeasured': coords_notmeasured,

            # mislabeling
            'coords_mislabeled_predunmeas': coords_mislabeled_predunmeas,
            'coords_mislabeled_predall': coords_mislabeled_predall

            # # gp pred unmeas & all
            # 'gp_predictions_unmeas': gp_predictions_unmeas,
            # 'gp_probabilities_unmeas': gp_probabilities_unmeas,
            # 'gp_confidences_unmeas': gp_confidences_unmeas,
            # 'gp_uncertainties_unmeas': gp_uncertainties_unmeas,
            # 'gp_predictions_all': gp_predictions_all,
            # 'gp_probabilities_all': gp_probabilities_all,
            # 'gp_confidences_all': gp_confidences_all,
            # 'gp_uncertainties_all': gp_uncertainties_all,

            # # gp uncertainty normalized, cdfs
            # 'gp_unc_01s': gp_unc_01s,
            # 'gp_unc_norms': gp_unc_norms,
            # 'gp_unc_cdfs': gp_unc_cdfs,

            # # edge weighted normalized, cdfs
            # 'unmeas_edges_ints': unmeas_edges_ints,
            # 'ew_unc_01s': ew_unc_01s,
            # 'ew_unc_norms': ew_unc_norms,
            # 'ew_unc_cdfs': ew_unc_cdfs,

            # # edge ratio weighted normalized, cdfs
            # 'unmeas_edges_ratios': unmeas_edges_ratios,
            # 'rw_unc_01s': rw_unc_01s,
            # 'rw_unc_norms': rw_unc_norms,
            # 'rw_unc_cdfs': rw_unc_cdfs,

            # # meas_distance
            # 'meas_distances': meas_distances,
            # 'meas_dist_norms': meas_dist_norms,
            # 'meas_dist_cdfs': meas_dist_cdfs,

            # # cdf/idx/next_coords tracking
            # 'all_cdfs': all_cdfs,
            # 'all_idxs': all_idxs,
            # 'all_next_coords': all_next_coords,

            # # gp times and kernel prms
            # 'gp_fittime': gp_fittime,
            # 'gp_kernelprms': gp_kernelprms,
            # 'gp_predtime_unmeas': gp_predtime_unmeas,
            # 'gp_probtime_unmeas': gp_probtime_unmeas,
            # 'gp_predtime_all': gp_predtime_all,
            # 'gp_probtime_all': gp_probtime_all,

            # # prediction scores
            # 'f1s_weighted_predall': f1s_weighted_predall,
            # 'f1s_macro_predall': f1s_macro_predall,
            # 'precisions_weighted_predall': precisions_weighted_predall,
            # 'precisions_macro_predall': precisions_macro_predall,
            # 'recalls_weighted_predall': recalls_weighted_predall,
            # 'recalls_macro_predall': recalls_macro_predall
            }


        # Naming
        run_id  = num_exp_runs[0]  # what you already computed
        name_core = f"{fname}_{approach}_{args.campaign}_run{run_id}"
        out_pkl  = f"{pkl_path}/{name_core}.pkl"
        out_fig1 = f"{fig_path}/{name_core}_accuracy_nummeas.png"
        out_fig2 = f"{fig_path}/{name_core}_accuracy_percmeas.png"

        #out_pkl = f'{pkl_path}/{fname}_{approach}_{exp_run}.pkl'
        with open(out_pkl, 'wb') as f:
            pickle.dump(data, f)
        print(f"\npickled data saved: {out_pkl}\n",flush=True)
        print(f"\nMaking accuracy plots\n", flush=True)

        ####################################
        # Accuracy plots
        ####################################
        plt.figure()
        plt.plot(data['num_measurements'], data['accuracies'], label='accuracy (combined labels / ground truth)', alpha=0.5)
        plt.plot(data['num_measurements'], data['accuracies_gpgt'], label='accuracy (model / best model)', alpha=0.5)
        plt.ylim((0,1))
        plt.xlim((data['num_measurements'][0],data['num_measurements'][-1]))
        plt.xlabel('Num. measurements'); plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.title(f"{fname} - {approach} (run: {exp_run})")
        plt.savefig(out_fig1)

        plt.figure()
        plt.plot(data['percent_measured'], data['accuracies'], label='accuracy (combined labels / ground truth)', alpha=0.5)
        plt.plot(data['percent_measured'], data['accuracies_gpgt'], label='accuracy (model / best model)', alpha=0.5)
        plt.ylim((0,1))
        plt.xlim((0.3,data['percent_measured'][-1]))
        plt.xlabel('% total measurements'); plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.title(f"{fname} - {approach} (run: {exp_run})")
        plt.savefig(out_fig2)

        time.sleep(3)
        plt.close('all')

        print(f"\nFinished {approach} exp {exp_run_num}\n\ttime: {(time.time() - exp_loop_start_time) / 60} minutes\n", flush=True)

    print(f"\n\nFinished {len(num_exp_runs)} {approach} experiments\n\ttotal time: {(time.time() - exp_loop_start_time) / 60} minutes\n\taverage time per experiment: {(time.time() - exp_loop_start_time) / 60 / max(len(num_exp_runs),1)} minutes\n\n", flush=True)

    print(f'Total elapsed time: {(time.time() - script_start_time) / 60} min.')

if __name__ == "__main__":
    main()
