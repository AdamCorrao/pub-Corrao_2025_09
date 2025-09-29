import sys, os, copy, time, json, pickle, ast, cv2, subprocess
#os.environ['OMP_NUM_THREADS'] = '1'
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
from tqdm.notebook import trange, tqdm
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

mpl.rcParams['mathtext.default'] = 'regular'

def get_repo_root() -> Path:
    # 1) If running inside a Git clone, ask Git
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        return Path(root)
    except Exception:
        pass
    # 2) Optional: allow an override via env var
    if os.getenv("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).expanduser().resolve()
    # 3) Fallback: current working directory
    return Path.cwd()

####################################################################################
# Define paths, create dirs if needed

root_dir = get_repo_root()

path = root_dir / "Data" / "simulatedwafer_datasets"
simexp_path = root_dir / "SimulatedExperiments"
fig_path = simexp_path / "figures"
pkl_path = simexp_path / "pkls"
png_path = simexp_path / "pngs"
movie_path = simexp_path / "movies"
snapshot_path = png_path / "campaign_snapshots"

# complex wafer
complex_zipfile = 'ds_AlLiFe_complex_21Sep2024_12-04-04.zip'
complex_file = 'ds_AlLiFe_complex_21Sep2024_12-04-04.nc'

####################################################################################

# Class / functions
class ExperimentManager:
    def __init__(self, ground_truth: xr.Dataset):
        """
        ExperimentManager assumes a circlular wafer and will exclude points outside radius

        Parameters
        ----------
        ground_truth : Xarray Dataset
            Dataset holding diffraction data and x,y coordinates.

        resolution : float
            scale to quantize the space on.
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

        # For testing interpolation behavior, identifying issues at edges
        # if np.isnan(interpolated_value).any():
        #     print('Got nans - adding -1')
        #     interpolated_value = np.full(self._ground_truth[dataarray].shape[2], -1)

        return interpolated_value

    def _get_iq_at_coordinates(self, x_coord, y_coord):
        """ given x and y, return a tuple of numpy arrays (q, Iq) """
        interpolated_value = self._ground_truth['iq'].interp({"x": x_coord, "y": y_coord}).data

        if np.isnan(interpolated_value).any():
            raise ValueError(f'x,y input ({x_coord}, {y_coord}) is outside sample bounds')

        # For testing interpolation behavior, identifying issues at edges
        # if np.isnan(interpolated_value).any():
        #     print('Got nans - adding -1')
        #     interpolated_value = np.full(self._ground_truth[dataarray].shape[2], -1)

        return self._q, interpolated_value

    def get_random_unmeasured_points(self, num_points):
        rng = np.random.default_rng()
        random_indices = rng.choice(self._coords_notmeasured.shape[0], num_points, replace=False) # replace=False prevents duplicates
        random_points = self._coords_notmeasured[random_indices]
        return random_points

    def measure(self, xy):
        """ collects data from coordinates (x,y), adds to 'sample' archive """
        x, y = xy
        iq_value = self._get_value_at_coordinates(x, y, dataarray='iq') # this will raise an errrr if Nan returned (outside bounds)

        # combining functionality from do_measurement
        if np.any(np.all(self._coords_valid == xy, axis=1)):
            # append x,y to _coords_measured
            self._coords_measured.append((xy))
            # remove x,y from _coords_notmeasured
            self._coords_notmeasured = self._coords_notmeasured[~np.all(self._coords_notmeasured == xy, axis=1)] # boolean mask for removing xy

        else:
            print(f"{xy} not a discretized point - data collected but NOT recorded in _coords_measured and NOT removed from _coords_notmeasured")

        # Update sample dataset
        # all we need is x, y, and intensity
        new_sample = xr.Dataset(
            {
                'iq': (('index', 'intensity'), [iq_value]),
            },
            coords={
                'x': (('index',), [x]),
                'y': (('index',), [y])
            }
        )

        self._seensamples.append(new_sample) # initialized as an empty list in the init


    def _plot_ground_truth(self, colormap = 'Set2', marker='s',marker_size=20, figsize=(10,6), alpha=0.5, legend_on=True, tight_layout=True, title='auto'):
        colors = plt.get_cmap(colormap)
        cmap = mcolors.ListedColormap(colors.colors[:len(set(self._ground_truth.ground_truth_labels.data))])
        # Step 2: Use BoundaryNorm to ensure that the colors are fixed to the labels
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(set(self._ground_truth.ground_truth_labels.data)), 1), ncolors=cmap.N)

        # Plot the data
        if figsize:
            plt.figure(figsize=(figsize[0],figsize[1]))
        else:
            plt.figure()
        scatter = plt.scatter(self._ground_truth.coords_valid.data[:,0], self._ground_truth.coords_valid.data[:,1], c=self._ground_truth.ground_truth_labels.data, cmap=cmap, norm=norm, marker=marker, s=marker_size, alpha=alpha)
        plt.xlabel('x')
        plt.ylabel('y')

        if title:
            if title == 'auto':
                plt.title(f'Ground truth')

            elif isinstance(title, str):
                plt.title(title)

        # Create the custom legend
        if legend_on is True:
            legend_elements = []
            for name, num in zip(self._label_names,self._label_nums):
                color = cmap(num)
                legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, label=name))

            # Add the legend outside the plot
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1), loc='upper left')

        if tight_layout is True:
            plt.tight_layout()

    def _get_ground_truth_labels(self):
        return self._ground_truth.ground_truth_labels.data

    # look up python descriptor method
    @property
    def sample(self):
        if len(self._seensamples) == 0:
            return xr.Dataset(
                {
                    'iq': (('index', 'intensity'), np.empty((0, self._ground_truth['iq'].shape[2]))),
                },
                coords={
                    'x': (('index',), []),
                    'y': (('index',), [])
                },
                attrs={
                    'num_measurements': 0}
            )

        # Concatenate new sample to the existing sample dataset
        sample = xr.concat(self._seensamples, dim='index')
        sample.attrs['num_measurements'] = len(self._seensamples)
        sample.attrs['Q'] = self._q

        return sample


## Functions
# get distances from not measured to measured points (input to wrapper function)
def measurement_dist_calc(measured_coords, notmeasured_coords):
    distances = distance_matrix(notmeasured_coords, measured_coords) # inputs are 2D arrays of points not measured and points measured
    minimum_distances = np.array([i.min() for i in distances])
    return minimum_distances

# fit some inputs (e.g., distances from kmeans) to a polynomial, returns distances predicted for input coords
def compute_pred_dist(min_dist, measured_coords, notmeasured_coords, initial_guess=(1, 1, 1, 1, 1, 1), print_polyparams=False):
    """
    min_dist: list or 1D nparray of minimum distance from each measured point to its nearest cluster
    """
    x = np.array(measured_coords)[:,0]
    y = np.array(measured_coords)[:,1]
    meas_coords = np.column_stack((x,y))

    def poly_func(meas_coords, a, b, c, d, e, f):
        return a*meas_coords[:,0]**2 + b*meas_coords[:,1]**2 + c*meas_coords[:,0]*meas_coords[:,1] + d*meas_coords[:,0] + e*meas_coords[:,1] + f

    params, _ = curve_fit(poly_func, meas_coords, min_dist, p0=initial_guess) # fit poly_func to xy coords, min_dist and get poly coefficients
    if print_polyparams is True:
        print(f"polynomial parameters: {params}")

    # Predicted distances using the fitted polynomial coefficients
    predicted_distances = poly_func(notmeasured_coords, *params)

    return predicted_distances

##############################################################################################################################
# Functions below here can be uniformly used with any input array of "distances" (i.e., both measurement- and kmeans-derived)
##############################################################################################################################

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
            dist = np.ones_like(dist) / len(dist)  # Set to a uniform distribution

    return dist

def get_cdf(dist):
    cdf = np.cumsum(dist) # cumulative distribution function
    return cdf

def sample_from_numberline(cdf, num_suggested_points, side='right'):
    r = np.random.rand(num_suggested_points) # random value(s) from 0 - 1
    idx = np.searchsorted(cdf, r, side=side) # index (or indices) corresponding to r value(s)
    return idx

def get_next_coords(notmeasured_coords, idx):
    next_coords = np.array([notmeasured_coords[i] for i in idx])
    return next_coords

# Wrapper function to get next coords with default behaviors
def suggest_next_coords(distances, notmeasured_coords, sigma=2, dist_power_scale=1, num_suggested_points=1,
                        display_distances=False, display_dist=False, display_cdf=False, display_next_coords=False, # optional plotting
                        disp_marker='s', disp_markersize=3, colormap='viridis',disp_figsize=(8,5)): # plotting args

    #predicted_distances = compute_pred_dist(experiment=experiment, min_dist=min_dist, coords_to_predict=coords_to_predict)
    dist = dist_transform_normalize(distances, sigma=sigma, dist_power_scale=dist_power_scale)
    cdf = get_cdf(dist)
    idx = sample_from_numberline(cdf, num_suggested_points=num_suggested_points)
    next_coords = get_next_coords(notmeasured_coords, idx=idx)

    # optional plotting
    if display_distances:
        if disp_figsize:
            plt.figure(figsize=(disp_figsize[0],disp_figsize[1]))
        else:
            plt.figure()

        plt.scatter(notmeasured_coords[:,0], notmeasured_coords[:,1], c=distances, cmap=colormap, marker=disp_marker, s=disp_markersize)
        plt.colorbar(label='input distances')
        plt.title('Input distances')

    if display_dist:
        if disp_figsize:
            plt.figure(figsize=(disp_figsize[0],disp_figsize[1]))
        else:
            plt.figure()

        plt.scatter(notmeasured_coords[:,0], notmeasured_coords[:,1], c=dist, cmap=colormap, marker=disp_marker, s=disp_markersize)
        plt.colorbar(label='Calculated distances (normalized)')
        plt.title('Calculated distances (normalized)')

    if display_cdf:
        if disp_figsize:
            plt.figure(figsize=(disp_figsize[0],disp_figsize[1]))
        else:
            plt.figure()

        plt.plot(cdf)
        plt.title('CDF')

    if display_next_coords:
        if disp_figsize:
            plt.figure(figsize=(disp_figsize[0],disp_figsize[1]))
        else:
            plt.figure()

        plt.scatter(notmeasured_coords[:,0], notmeasured_coords[:,1], c=dist, cmap=colormap, marker=disp_marker, s=disp_markersize)
        plt.colorbar(label='Calculated distances (normalized)')
        plt.scatter(next_coords[:,0], next_coords[:,1], marker='x', color='red')
        plt.title('Suggested measurement(s)')

    return next_coords

##################################
# GP model evaluation functions
##################################

# Confusion matrix between ground truth and predicted labels
def match_labels(ground_truth_labels, predicted_labels):
    max_label = max(ground_truth_labels.max(), predicted_labels.max()) + 1 # returns largest of positional arguments, adds 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int) # make empty confusion matrix

    for true, pred in zip(ground_truth_labels, predicted_labels): # populates confusion matrix with occurences
        confusion_matrix[true, pred] += 1

    # Solve the assignment problem - minimizes the cost of label mismatch. Finds best 1-to-1 assignment between rows (true labels) and columns (predicted labels)
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    # Create a mapping from predicted labels to true labels
    label_mapping = dict(zip(col_ind, row_ind)) # describes mapping of predicted label to true labels (e.g., 2->0, 1->1, 0->2)
    predicted_labels_mapped = np.array([label_mapping[p] for p in predicted_labels])

    return predicted_labels_mapped

def get_pred_accuracy(ground_truth_labels, aligned_predicted_labels, print_accuracy_stats=False):

    if print_accuracy_stats:
        count_match = 0
        count_notmatch = 0

        for label, predicted_label in zip(ground_truth_labels, aligned_predicted_labels):
            if label == predicted_label:
                count_match += 1
            else:
                count_notmatch += 1

        print(f"label matches: {count_match} / {len(ground_truth_labels)}\nnum not matching: {count_notmatch} / {len(ground_truth_labels)}\nAccuracy: {count_match / len(ground_truth_labels) * 100}")

    accuracy = np.mean(aligned_predicted_labels == ground_truth_labels)
    #print(f'Accuracy: {accuracy * 100:.2f}%')

    return accuracy

# labels and coors must have same order or indices will not correspond
def get_mislabeled_coords(ground_truth_labels, aligned_predicted_labels, valid_coords):
    mislabeled_indices = np.where(ground_truth_labels != aligned_predicted_labels)[0]
    mislabeled_coords = valid_coords[mislabeled_indices]

    return mislabeled_coords
def geometric_spacing(xmin, xmax, ymin, ymax, n, seen_list = [], cen_xy = (0,0), rmax=-1):
    #print (f'making for 2^{n}+1 = {2**n+1}')
    nuse = int(2**n+1)
    xlist = np.linspace(xmin, xmax, nuse)
    ylist = np.linspace(ymin, ymax, nuse)
    my_list = []

    def dist_good(xy, return_val = False):
        if rmax < 0:
            return True
        else:
            dx = abs(cen_xy[0] - xy[0])
            dy = abs(cen_xy[1] - xy[1])

            if return_val:
                return np.sqrt(dx**2. + dy**2.)

            if np.sqrt(dx**2. + dy**2.) <= rmax:
                return True
            else:
                return False

    check_seen_list = []
    for xy in np.array(seen_list).T:
        check_seen_list.append(tuple(xy))

    for y in ylist:
        for x in xlist:
            coords = [np.round(x,2),np.round(y,2)]

            #check distance
            if dist_good(coords):
                if tuple(coords) not in check_seen_list:
                    my_list.append(coords)
            else:
                #print (f"uh oh... {coords}")
                #print (f"{dist_good(coords, return_val=True)}")
                pass
    return np.array(my_list).T

def geoseries(xmin, xmax, ymin, ymax, nmax, cen_xy = (0,0), rmax=-1):
    """ Creates a geometric series from xmin->xmax, and ymin->ymax,
    with nmax total iterations on the geometric series loop.
    If you provide a tuple center of the wafer for cen_xy, and
    a maximum radius with rmax, you will exclude points beyond that."""

    this = geometric_spacing(xmin,xmax,ymin,ymax,0, cen_xy = cen_xy, rmax=rmax)

    for nis in range(nmax):
        this2 = geometric_spacing(xmin,xmax,ymin,ymax,nis+1, seen_list=this, cen_xy = cen_xy, rmax=rmax)
        if len(this) >0:
            this = np.concatenate([this.T, this2.T]).T
        else:
            this = this2
    return this

def geoseries_to_N(xmin, xmax, ymin, ymax, N, cen_xy = (0,0), rmax=-1):
    """ calculates the geometric series out to N points, excludes others """
    go_on = False

    use_n = 0
    while not go_on:
        this_series = geoseries(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, nmax=use_n, cen_xy = cen_xy, rmax=rmax)
        use_n += 1
        if len(this_series.T) >= N:
            go_on = True
    return this_series.T[:N].T



def detect_edges(coords, labels, threshold):
    # Initialize arrays for results
    edge_detect = np.zeros(len(coords), dtype=bool)
    edge_ratio = np.zeros(len(coords), dtype=float)

    # Loop over each coordinate and find its neighbors
    for i, coord in enumerate(coords):
        # Calculate distances to all other coordinates
        distances = np.linalg.norm(coords - coord, axis=1)

        # Find neighbors (excluding the point itself)
        neighbor_mask = (distances > 0) & (distances <= threshold)
        neighbor_indices = np.where(neighbor_mask)[0]

        if len(neighbor_indices) > 0:
            # Compare labels
            neighbor_labels = labels[neighbor_indices]
            current_label = labels[i]

            # Check if any neighbor has a different label
            different_neighbors = neighbor_labels != current_label
            edge_detect[i] = np.any(different_neighbors)

            # Calculate the ratio of different labels
            edge_ratio[i] = np.sum(different_neighbors) / len(neighbor_indices)

    return edge_detect, edge_ratio

def detect_edges_with_distances(distances, labels, threshold):
    # Initialize arrays for results
    edge_detect = np.zeros(len(labels), dtype=bool)
    edge_ratio = np.zeros(len(labels), dtype=float)

    # Loop over each label and its corresponding distances
    for i in range(len(labels)):
        # Find neighbors based on precomputed distances
        neighbor_mask = (distances[i] > 0) & (distances[i] <= threshold)
        neighbor_indices = np.where(neighbor_mask)[0]

        if len(neighbor_indices) > 0:
            # Compare labels
            neighbor_labels = labels[neighbor_indices]
            current_label = labels[i]

            # Check if any neighbor has a different label
            different_neighbors = neighbor_labels != current_label
            edge_detect[i] = np.any(different_neighbors)

            # Calculate the ratio of different labels
            edge_ratio[i] = np.sum(different_neighbors) / len(neighbor_indices)

    return edge_detect, edge_ratio


def sample_and_remove(array):
    if len(array) == 0:
        raise ValueError("Array is empty. No more elements to sample.")

    # Randomly choose an index
    random_index = np.random.randint(len(array))

    # Get the randomly sampled coordinate pair
    sampled_coordinate = array[random_index]

    # Remove the sampled coordinate pair from the array
    array = np.delete(array, random_index, axis=0)

    return sampled_coordinate, array


def prob_norm(array):
    array_sum = array.sum()

    if array_sum != 0:
        array_norm = array / array_sum
    else:
        array_norm = np.ones_like(array) / len(array) # set to a uniform distribution

    return array_norm

def minmax_norm(array):
    array_norm = (array - array.min()) / (array.max() - array.min())

    return array_norm

#####################################
# Experiment below here
#####################################
script_start_time = time.time()
num_exp_runs = [0] # can provide a list of ints to run in a loop (e.g., np.arange(10), [0,1,2,3,4])

exp_loop_start_time = time.time()

for exp_run_num in num_exp_runs:

    exp_start_time = time.time() # loop timing

    total_meas_todo = 805 # total measurements to do
    exp_run = 'seed49_mmm1_' + str(exp_run_num)
    approach = 'gpewunc' # string used in filenaming

    print(f'\n\n\n###############\nStarting {approach} experiment {exp_run_num}!\n###############\n\n\n', flush=True)

    # swap out datasets here
    zipfile = complex_zipfile
    file = complex_file

    if file not in os.listdir(path):
        with ZipFile(os.path.join(path,zipfile), 'r') as zObject:
            zObject.extract(file, path=path)
        zObject.close()
    ground_truth = xr.open_dataset(os.path.join(path,file)) # read in dataset

    experiment = ExperimentManager(ground_truth) # initialize experiment

    # Instantiate kmeans
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Define the gaussian process kernel
    kernel = C(1.0, (1e-3, 10)) * RBF(1.0, (1e-3, 30))
    #kernel = C(1.0, (1e-3, 10)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3,30), nu=0.5) # nu 1.5 = once diff. func, 2.5 = twice diff. func.

    # Instantiate the classifier
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1, n_restarts_optimizer=5)

    # get ground_truth labels
    ground_truth_labels = experiment._get_ground_truth_labels()

    # model labels (from GP prediction after training on all data)
    gp_df = pd.read_csv(os.path.join(path, 'gp_prediction_onalldata.csv'))
    ground_truth_labels_gp = gp_df.predicted_labels.values

    # Geometric series stuff
    xmin = experiment._coords_valid[:,0].min()
    xmax = experiment._coords_valid[:,0].max()
    ymin = experiment._coords_valid[:,1].min()
    ymax = experiment._coords_valid[:,1].max()
    cen_xy = experiment._ground_truth.attrs['shape_center']
    rmax = ymax - 0.4 # 1 pixel in from the edge for this series
    #print(f"xmin: {xmin} xmax: {xmax}\nymin: {ymin} ymax: {ymax}")
    geox, geoy = geoseries(xmin = xmin, xmax = xmax,
                    ymin = ymin, ymax = ymax,
                    nmax = 6, #number of geometric iterations (5=797, 6=3209, 7=12853)
                    cen_xy=cen_xy,rmax=ymax)
    geotomeas = np.column_stack((geox,geoy)) # this is our list to measure from, in order

    # series of points
    geotomeas_1 = geotomeas[0:13]
    geotomeas_2 = geotomeas[13:49]
    geotomeas_3 = geotomeas[49:197]
    geotomeas_4 = geotomeas[197:797]
    geotomeas_5 = geotomeas[797:3209]

    ###################################
    # Bookkeeeping
    ###################################
    measured_labels = []
    num_measurements = []

    # 2 diff accuracies
    accuracies = [] # measured (N) + model (Ntot - N) / measured (Ntot) | meas + predicted (unmeas) / gt_labels
    accuracies_gpgt = [] # model(N) + model (Ntot - N) / Model (Ntot) | | predict all / gpgt_labels

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

    # prediction on all (gt labels) - is this the true "model?"
    gp_predictions_all = [] # corresponds to accuracies_gpgt
    gp_probabilities_all = []
    gp_confidences_all = []
    gp_uncertainties_all = []

    # labels
    all_combined_labels = [] # measured labels (gt) combined with unmeas GP-predicted labels (corresponding to accuracies) - this is meas + model

    coords_measured = []
    coords_notmeasured = []
    coords_mislabeled_predunmeas = [] # mislabeled coords from combined labels (meas + model / meas)
    coords_mislabeled_predall = [] # mislabeled coords from pred all labels (model + model / model)

    # bookkeeping of measurement suggestions (from gp pred unmeas)
    # gp unc
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


    # initial measurements
    init_meas = np.vstack((geotomeas_1,geotomeas_2)) # assuming we seed with 49 pts

    for i in range(init_meas.shape[0]):
        # measure first N points in geoseries (find nearest in valid coords)
        # Update: find nearest in _coords_notmeasured to avoid duplicate measurements
        # geodist = np.linalg.norm(experiment._coords_valid - init_meas[i], axis=1)
        # closest_idx = np.argmin(geodist)
        # experiment.measure((experiment._coords_valid[:,0][closest_idx], experiment._coords_valid[:,1][closest_idx]))

        geodist = np.linalg.norm(experiment._coords_notmeasured - init_meas[i], axis=1)
        closest_idx = np.argmin(geodist)
        experiment.measure((experiment._coords_notmeasured[:,0][closest_idx], experiment._coords_notmeasured[:,1][closest_idx]))

    print(f"\nInitial {init_meas.shape[0]} measurements done\n", flush=True)

    # Measurement loop
    for i in range(experiment.sample.attrs['num_measurements'], total_meas_todo):
        print(f'\n{approach} exp {exp_run_num}: starting measurement {i + 1} / {total_meas_todo}...\n', flush=True)

        #############################
        # bookkeeping measurements
        #############################
        coords_measured.append(np.array(experiment._coords_measured))
        coords_notmeasured.append(experiment._coords_notmeasured)
        num_measurements.append(experiment.sample.attrs['num_measurements'])

        ###########################################
        # get labels from ground truth directly (no clustering)
        ###########################################
        measured_indices = []
        # Loop through each x, y pair in the measured array
        for xy in np.array(experiment._coords_measured):
            # Find the index in the second array (valid) where both elements match the current xy pair
            match = np.where((experiment._coords_valid == xy).all(axis=1))[0]

            # If there's a match, add the index to the list
            if match.size > 0:
                measured_indices.append(match[0])
            else:
                raise ValueError(f'measured point {xy} not found in valid coords - inspect for issue')

        # Convert the indices list to an array
        measured_indices = np.array(measured_indices)

        # Get the corresponding labels from ground_truth_labels using the indices - will be from clustering in real exp
        labels = ground_truth_labels[measured_indices]
        measured_labels.append(labels)


        ################################################
        # GP fitting
        ################################################
        #if experiment.sample.attrs['num_measurements'] % 1 == 0 or not gp_fit_called:
        start_time = time.time()
        gpc.fit(np.array(experiment._coords_measured), labels)
        gp_fittime.append((time.time() - start_time))
        gp_kernelprms.append(gpc.kernel_.get_params()['kernels'])

        ################################################
        # GP prediction - unmeasured points
        ################################################
        # Predict the most likely label for each point
        start_time = time.time()
        predicted_labels = gpc.predict(experiment._coords_notmeasured)
        gp_predtime_unmeas.append((time.time() - start_time))

        # # Predict probabilities for all data
        start_time = time.time()
        #probabilities = gpc.predict_proba(experiment._coords_valid)
        probabilities = gpc.predict_proba(experiment._coords_notmeasured)
        gp_probtime_unmeas.append((time.time() - start_time))
        confidence = np.max(probabilities, axis=1) # we want to then pick the point with lowest confidence!
        uncertainty = entropy(probabilities.T) # entropy quantifies uncertainty by looking at how probability is distributed amongst the classes

        gp_predictions_unmeas.append(predicted_labels) # labels only for unmeasured points
        gp_probabilities_unmeas.append(probabilities) # only for unmeasured points - full prob distribution (n_samples, n_classes)
        gp_confidences_unmeas.append(confidence)
        gp_uncertainties_unmeas.append(uncertainty) # only for unmeasured points

        # combining known / measured labels + GP-predicted labels, aligned along valid coords
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
                raise ValueError(f'coord {coord} not found in coords_measured or coords_notmeasured - inspect this critical error!')

        combined_labels = np.array(combined_labels)
        all_combined_labels.append(combined_labels)
        mislabeled_coords = get_mislabeled_coords(ground_truth_labels, combined_labels, experiment._coords_valid)
        coords_mislabeled_predunmeas.append(mislabeled_coords)

        # Get prediction score
        accuracy = get_pred_accuracy(ground_truth_labels, combined_labels)
        accuracies.append(accuracy)


        ################################################
        # GP prediction - all points
        ################################################
        # Predict the most likely label for each point
        start_time = time.time()
        predicted_labels = gpc.predict(experiment._coords_valid)
        gp_predtime_all.append((time.time() - start_time))

        # # Predict probabilities for all data
        start_time = time.time()
        probabilities = gpc.predict_proba(experiment._coords_valid)
        gp_probtime_all.append((time.time() - start_time))
        confidence = np.max(probabilities, axis=1) # we want to then pick the point with lowest confidence!
        uncertainty = entropy(probabilities.T) # entropy quantifies uncertainty by looking at how probability is distributed amongst the classes

        gp_predictions_all.append(predicted_labels)
        gp_probabilities_all.append(probabilities) #  full prob distribution (n_samples, n_classes)
        gp_confidences_all.append(confidence)
        gp_uncertainties_all.append(uncertainty)

        #all_labels_cv.append(np.array(predicted_labels)) # is this already an array?
        mislabeled_coords = get_mislabeled_coords(ground_truth_labels_gp, predicted_labels, experiment._coords_valid) # mislabeled relative to GP trained on all data
        coords_mislabeled_predall.append(mislabeled_coords)

        # Get prediction score
        accuracy = get_pred_accuracy(ground_truth_labels_gp, predicted_labels) # gp ground truth

        f1_w = f1_score(ground_truth_labels_gp, predicted_labels, average='weighted')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)
        precision_w = precision_score(ground_truth_labels_gp, predicted_labels, average='weighted')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)
        recall_w = recall_score(ground_truth_labels_gp, predicted_labels, average='weighted')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)

        f1_m = f1_score(ground_truth_labels_gp, predicted_labels, average='macro')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)
        precision_m = precision_score(ground_truth_labels_gp, predicted_labels, average='macro')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)
        recall_m = recall_score(ground_truth_labels_gp, predicted_labels, average='macro')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)

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

        # Edge detection (uses measured + predicted labels)
        edge_detect, edge_ratio = detect_edges_with_distances(experiment._dist_btwn_coords,
                                            combined_labels,
                                            threshold=experiment._ground_truth.attrs['resolution'] * 1.5)
        #print(f'num edges: {len([i for i in edge_detect if i == True])}')

        all_indices = np.arange(experiment._coords_valid.shape[0])
        unmeasured_indices = np.setdiff1d(all_indices, measured_indices)
        unmeasured_labels = combined_labels[unmeasured_indices]

        unmeas_edges = edge_detect[unmeasured_indices] # boolean array
        unmeas_edges_int = unmeas_edges.astype(int) # turn boolean array to 0 or 1 (off/on edge)
        unmeas_edges_ratio = edge_ratio[unmeasured_indices] # floats (0 - 1) where 1 = all neighbors different, 0 = all neighbors same

        # Normalization, getting cdfs | updated Jan 2025 for simplicity (not using dist_transform_normalize now, power scaling unnecessary)
        gp_unc = gp_uncertainties_unmeas[-1]

        # normalize (sum to 1), calc cdf | not using because this is often featureless and cannot be combined with edge
        # gp_unc_norm = prob_norm(gp_unc)
        # gp_unc_cdf = get_cdf(gp_unc_norm)

        # normalize (0 to 1), normalize (sum to 1), calc cdf
        gp_unc_01 = minmax_norm(gp_unc) # for plotting uncertainty / movies
        gp_unc_norm = prob_norm(gp_unc_01)
        gp_unc_cdf = get_cdf(gp_unc_norm)

        # normalize (0 to 1), power scale (to the 8th), normalize (sum to 1), calc cdf
        # gp_unc_norm01 = minmax_norm(gp_unc)
        # gp_unc_norm01p = gp_unc_norm01**8
        # gp_unc_norm01p_norm = prob_norm(gp_unc_norm01p)
        # gp_unc_norm01p_cdf = get_cdf(gp_unc_norm01p_norm)

        # edge weighted uncertainty
        ew_unc_01 = minmax_norm(minmax_norm(gp_unc) + minmax_norm(unmeas_edges_int)) # normalize both 0 to 1, add (normalize to 1 again, just for plotting)
        ew_unc_norm = prob_norm(ew_unc_01) # normalize summed array to sum to 1
        ew_unc_cdf = get_cdf(ew_unc_norm)

        # edge ratio weighted uncertainty
        rw_unc_01 = minmax_norm(minmax_norm(gp_unc) + minmax_norm(unmeas_edges_ratio)) # normalize both 0 to 1, add (normalize to 1 again, just for plotting)
        rw_unc_norm = prob_norm(rw_unc_01) # normalize summed array to sum to 1
        rw_unc_cdf = get_cdf(rw_unc_norm)


        # proximity / meas distances - gauss-like and normalized (sum to 1) | not normalized 0 to 1
        meas_distance = measurement_dist_calc(measured_coords=experiment._coords_measured, notmeasured_coords=experiment._coords_notmeasured)
        meas_dist_norm = dist_transform_normalize(meas_distance, sigma=2, dist_power_scale=2, normalize=True)
        meas_cdf = get_cdf(meas_dist_norm)


        # idx, next coord that is driving measurement
        cdf = ew_unc_cdf
        idx = sample_from_numberline(cdf, num_suggested_points=1, side='right')
        next_coords = get_next_coords(notmeasured_coords=experiment._coords_notmeasured, idx=idx)

        all_cdfs.append(cdf)
        all_idxs.append(idx)
        all_next_coords.append(next_coords)

        # bookkeeping that will be relevant for plotting
        gp_unc_01s.append(gp_unc_01)
        gp_unc_norms.append(gp_unc_norm)
        gp_unc_cdfs.append(gp_unc_cdf)

        # edge weighted
        unmeas_edges_ints.append(unmeas_edges_int)
        ew_unc_01s.append(ew_unc_01)
        ew_unc_norms.append(ew_unc_norm)
        ew_unc_cdfs.append(ew_unc_cdf)

        # edge ratio weighted
        unmeas_edges_ratios.append(unmeas_edges_ratio)
        rw_unc_01s.append(rw_unc_01)
        rw_unc_norms.append(rw_unc_norm)
        rw_unc_cdfs.append(rw_unc_cdf)

        # meas_distance
        meas_distances.append(meas_distance)
        meas_dist_norms.append(meas_dist_norm)
        meas_dist_cdfs.append(meas_cdf)

        # #######################################
        # # Random sampling of geometric series
        # #######################################

        # if i in np.arange(0,13):
        #     coord_to_meas, geotomeas_1 = sample_and_remove(geotomeas_1)

        # elif i in np.arange(13,49):
        #     coord_to_meas, geotomeas_2 = sample_and_remove(geotomeas_2)

        # elif i in np.arange(49,197):
        #     coord_to_meas, geotomeas_3 = sample_and_remove(geotomeas_3)

        # elif i in np.arange(197,797):
        #     coord_to_meas, geotomeas_4 = sample_and_remove(geotomeas_4)

        # elif i in np.arange(797,3209):
        #     coord_to_meas, geotomeas_5 = sample_and_remove(geotomeas_5)

        # # find closest coordinate, do measurement

        # #geodist = np.linalg.norm(experiment._coords_valid - init_meas[i], axis=1)
        # #closest_idx = np.argmin(geodist)
        # #experiment.measure((experiment._coords_valid[:,0][closest_idx], experiment._coords_valid[:,1][closest_idx]))

        # geodist = np.linalg.norm(experiment._coords_notmeasured - coord_to_meas, axis=1)
        # closest_idx = np.argmin(geodist)
        # experiment.measure((experiment._coords_notmeasured[:,0][closest_idx], experiment._coords_notmeasured[:,1][closest_idx]))

        #######################################
        # GP driven
        #######################################
        # suggestion-driven measurements
        for x,y in next_coords:
            experiment.measure((x,y))

        # random measurements
        # points_to_measure = experiment.get_random_unmeasured_points(1)
        # for x,y in points_to_measure:
        #     experiment.measure((x,y))


    print(f"\n{total_meas_todo} measurements done - starting end of run bookkeeping\n", flush=True)

    # End of run
    #############################
    # bookkeeping measurements
    #############################
    coords_measured.append(np.array(experiment._coords_measured))
    coords_notmeasured.append(experiment._coords_notmeasured)
    num_measurements.append(experiment.sample.attrs['num_measurements'])

    ###########################################
    # get labels from ground truth directly (no clustering)
    ###########################################
    measured_indices = []
    # Loop through each x, y pair in the measured array
    for xy in np.array(experiment._coords_measured):
        # Find the index in the second array (valid) where both elements match the current xy pair
        match = np.where((experiment._coords_valid == xy).all(axis=1))[0]

        # If there's a match, add the index to the list
        if match.size > 0:
            measured_indices.append(match[0])
        else:
            raise ValueError(f'measured point {xy} not found in valid coords - inspect for issue')

    # Convert the indices list to an array
    measured_indices = np.array(measured_indices)

    # Get the corresponding labels from ground_truth_labels using the indices - will be from clustering in real exp
    labels = ground_truth_labels[measured_indices]
    measured_labels.append(labels)


    ################################################
    # GP fitting
    ################################################
    #if experiment.sample.attrs['num_measurements'] % 1 == 0 or not gp_fit_called:
    start_time = time.time()
    gpc.fit(np.array(experiment._coords_measured), labels)
    gp_fittime.append((time.time() - start_time))
    gp_kernelprms.append(gpc.kernel_.get_params()['kernels'])

    ################################################
    # GP prediction - unmeasured points
    ################################################
    # Predict the most likely label for each point
    start_time = time.time()
    predicted_labels = gpc.predict(experiment._coords_notmeasured)
    gp_predtime_unmeas.append((time.time() - start_time))

    # # Predict probabilities for all data
    start_time = time.time()
    #probabilities = gpc.predict_proba(experiment._coords_valid)
    probabilities = gpc.predict_proba(experiment._coords_notmeasured)
    gp_probtime_unmeas.append((time.time() - start_time))
    confidence = np.max(probabilities, axis=1) # we want to then pick the point with lowest confidence!
    uncertainty = entropy(probabilities.T) # entropy quantifies uncertainty by looking at how probability is distributed amongst the classes

    gp_predictions_unmeas.append(predicted_labels) # labels only for unmeasured points
    gp_probabilities_unmeas.append(probabilities) # only for unmeasured points - full prob distribution (n_samples, n_classes)
    gp_confidences_unmeas.append(confidence)
    gp_uncertainties_unmeas.append(uncertainty) # only for unmeasured points

    # combining known / measured labels + GP-predicted labels, aligned along valid coords
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
            raise ValueError(f'coord {coord} not found in coords_measured or coords_notmeasured - inspect this critical error!')

    combined_labels = np.array(combined_labels)
    all_combined_labels.append(combined_labels)
    mislabeled_coords = get_mislabeled_coords(ground_truth_labels, combined_labels, experiment._coords_valid)
    coords_mislabeled_predunmeas.append(mislabeled_coords)

    # Get prediction score
    accuracy = get_pred_accuracy(ground_truth_labels, combined_labels)
    accuracies.append(accuracy)


    ################################################
    # GP prediction - all points
    ################################################
    # Predict the most likely label for each point
    start_time = time.time()
    predicted_labels = gpc.predict(experiment._coords_valid)
    gp_predtime_all.append((time.time() - start_time))

    # # Predict probabilities for all data
    start_time = time.time()
    probabilities = gpc.predict_proba(experiment._coords_valid)
    gp_probtime_all.append((time.time() - start_time))
    confidence = np.max(probabilities, axis=1) # we want to then pick the point with lowest confidence!
    uncertainty = entropy(probabilities.T) # entropy quantifies uncertainty by looking at how probability is distributed amongst the classes

    gp_predictions_all.append(predicted_labels)
    gp_probabilities_all.append(probabilities) #  full prob distribution (n_samples, n_classes)
    gp_confidences_all.append(confidence)
    gp_uncertainties_all.append(uncertainty)

    #all_labels_cv.append(np.array(predicted_labels)) # is this already an array?
    mislabeled_coords = get_mislabeled_coords(ground_truth_labels_gp, predicted_labels, experiment._coords_valid) # mislabeled relative to GP trained on all data
    coords_mislabeled_predall.append(mislabeled_coords)

    # Get prediction score
    accuracy = get_pred_accuracy(ground_truth_labels_gp, predicted_labels) # gp ground truth

    f1_w = f1_score(ground_truth_labels_gp, predicted_labels, average='weighted')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)
    precision_w = precision_score(ground_truth_labels_gp, predicted_labels, average='weighted')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)
    recall_w = recall_score(ground_truth_labels_gp, predicted_labels, average='weighted')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)

    f1_m = f1_score(ground_truth_labels_gp, predicted_labels, average='macro')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)
    precision_m = precision_score(ground_truth_labels_gp, predicted_labels, average='macro')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)
    recall_m = recall_score(ground_truth_labels_gp, predicted_labels, average='macro')  # or 'macro' to give equal weight to all classes (averages class-wise F1 scores)

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

    # Edge detection (uses measured + predicted labels)
    edge_detect, edge_ratio = detect_edges_with_distances(experiment._dist_btwn_coords,
                                        combined_labels,
                                        threshold=experiment._ground_truth.attrs['resolution'] * 1.5)
    #print(f'num edges: {len([i for i in edge_detect if i == True])}')

    all_indices = np.arange(experiment._coords_valid.shape[0])
    unmeasured_indices = np.setdiff1d(all_indices, measured_indices)
    unmeasured_labels = combined_labels[unmeasured_indices]

    unmeas_edges = edge_detect[unmeasured_indices] # boolean array
    unmeas_edges_int = unmeas_edges.astype(int) # turn boolean array to 0 or 1 (off/on edge)
    unmeas_edges_ratio = edge_ratio[unmeasured_indices] # floats (0 - 1) where 1 = all neighbors different, 0 = all neighbors same

    # Normalization, getting cdfs | updated Jan 2025 for simplicity (not using dist_transform_normalize now, power scaling unnecessary)
    gp_unc = gp_uncertainties_unmeas[-1]

    # normalize (sum to 1), calc cdf | not using because this is often featureless and cannot be combined with edge
    # gp_unc_norm = prob_norm(gp_unc)
    # gp_unc_cdf = get_cdf(gp_unc_norm)

    # normalize (0 to 1), normalize (sum to 1), calc cdf
    gp_unc_01 = minmax_norm(gp_unc) # for plotting uncertainty / movies
    gp_unc_norm = prob_norm(gp_unc_01)
    gp_unc_cdf = get_cdf(gp_unc_norm)

    # normalize (0 to 1), power scale (to the 8th), normalize (sum to 1), calc cdf
    # gp_unc_norm01 = minmax_norm(gp_unc)
    # gp_unc_norm01p = gp_unc_norm01**8
    # gp_unc_norm01p_norm = prob_norm(gp_unc_norm01p)
    # gp_unc_norm01p_cdf = get_cdf(gp_unc_norm01p_norm)

    # edge weighted uncertainty
    ew_unc_01 = minmax_norm(minmax_norm(gp_unc) + minmax_norm(unmeas_edges_int)) # normalize both 0 to 1, add (normalize to 1 again, just for plotting)
    ew_unc_norm = prob_norm(ew_unc_01) # normalize summed array to sum to 1
    ew_unc_cdf = get_cdf(ew_unc_norm)

    # edge ratio weighted uncertainty
    rw_unc_01 = minmax_norm(minmax_norm(gp_unc) + minmax_norm(unmeas_edges_ratio)) # normalize both 0 to 1, add (normalize to 1 again, just for plotting)
    rw_unc_norm = prob_norm(rw_unc_01) # normalize summed array to sum to 1
    rw_unc_cdf = get_cdf(rw_unc_norm)


    # proximity / meas distances - gauss-like and normalized (sum to 1) | not normalized 0 to 1
    meas_distance = measurement_dist_calc(measured_coords=experiment._coords_measured, notmeasured_coords=experiment._coords_notmeasured)
    meas_dist_norm = dist_transform_normalize(meas_distance, sigma=2, dist_power_scale=2, normalize=True)
    meas_cdf = get_cdf(meas_dist_norm)


    # idx, next coord that is driving measurement
    cdf = ew_unc_cdf
    idx = sample_from_numberline(cdf, num_suggested_points=1, side='right')
    next_coords = get_next_coords(notmeasured_coords=experiment._coords_notmeasured, idx=idx)

    all_cdfs.append(cdf)
    all_idxs.append(idx)
    all_next_coords.append(next_coords)

    # bookkeeping that will be relevant for plotting
    gp_unc_01s.append(gp_unc_01)
    gp_unc_norms.append(gp_unc_norm)
    gp_unc_cdfs.append(gp_unc_cdf)

    # edge weighted
    unmeas_edges_ints.append(unmeas_edges_int)
    ew_unc_01s.append(ew_unc_01)
    ew_unc_norms.append(ew_unc_norm)
    ew_unc_cdfs.append(ew_unc_cdf)

    # edge ratio weighted
    unmeas_edges_ratios.append(unmeas_edges_ratio)
    rw_unc_01s.append(rw_unc_01)
    rw_unc_norms.append(rw_unc_norm)
    rw_unc_cdfs.append(rw_unc_cdf)

    # meas_distance
    meas_distances.append(meas_distance)
    meas_dist_norms.append(meas_dist_norm)
    meas_dist_cdfs.append(meas_cdf)


    print(f"\nbookkeeping done - setting up dict for pkling data\n", flush=True)

    ####################################
    # Saving results
    ####################################

    # cv is for all coords, others are for only unmeasured points
    data = {
        'ground_truth_labels': ground_truth_labels,
        'ground_truth_labels_gp': ground_truth_labels_gp,
        'num_measurements': num_measurements,
        'percent_measured': (np.array(num_measurements) / ground_truth_labels.shape[0]) * 100,
        'accuracies': accuracies,
        'accuracies_gpgt': accuracies_gpgt,

        # labels
        'measured_labels': measured_labels,
        'all_combined_labels': all_combined_labels, # measured labels (gt) combined with unmeas GP-predicted labels (corresponding to accuracies) - this is meas + model
        'all_labels_predall': gp_predictions_all,

        # coords
        'coords_valid': experiment._coords_valid,
        'coords_measured': coords_measured,
        'coords_notmeasured': coords_notmeasured,

        # mislabeling
        'coords_mislabeled_predunmeas': coords_mislabeled_predunmeas, # mislabeled coords from combined labels (meas + model / meas)
        'coords_mislabeled_predall': coords_mislabeled_predall, # mislabeled coords from pred all labels (model + model / model)

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

        # meas_distance (not used, but may be informative)
        'meas_distances': meas_distances,
        'meas_dist_norms': meas_dist_norms,
        'meas_dist_cdfs': meas_dist_cdfs,

        # separate bookkeeping for cdf, idx, coords for measurement suggestion
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


    # pickling data
    if 'dansam1' in zipfile or 'dansam2' in zipfile:
        words = zipfile.split('_',-1)
        fname = '-'.join(words[:2])
    elif 'complex' in zipfile:
        fname = 'complex'
    elif 'simpleLR' in zipfile:
        fname = 'simpleLR'

    with open(f'{pkl_path}/{fname}_{approach}_{exp_run}.pkl', 'wb') as f:
        pickle.dump(data, f)

    print(f"\npickled data saved\n",flush=True)
    print(f"\nMaking accuracy plots\n", flush=True)

    ####################################
    # Accuracy plots
    ####################################
    plt.figure()
    plt.plot(data['num_measurements'], data['accuracies_gpgt'], label='accuracy (model / best model)', alpha=0.5)
    plt.ylim((0,1))
    plt.xlim((data['num_measurements'][0],data['num_measurements'][-1]))
    plt.xlabel('Num. measurements')
    plt.ylabel('Accuracy')
    plt.title(f"{fname} - {approach} (run: {exp_run})")
    plt.savefig(f'{fig_path}/{fname}_{approach}_{exp_run}_accuracy_nummeas.png')

    plt.figure()
    plt.plot(data['percent_measured'], data['accuracies_gpgt'], label='accuracy (model / best model)', alpha=0.5)
    plt.ylim((0,1))
    plt.xlim((0.3,data['percent_measured'][-1]))
    plt.xlabel('% total measurements')
    plt.ylabel('Accuracy')
    plt.title(f"{fname} - {approach} (run: {exp_run})")
    plt.savefig(f'{fig_path}/{fname}_{approach}_{exp_run}_accuracy_percmeas.png')

    time.sleep(3) # wait to close plots incase they are still being written
    plt.close('all')

    print(f"\nFinished {approach} exp {exp_run_num}\n\ttime: {(time.time() - exp_start_time) / 60} minutes\n", flush=True)

print(f"\n\nFinished {len(num_exp_runs)} {approach} experiments\n\ttotal time: {(time.time() - exp_loop_start_time) / 60} minutes\n\taverage time per experiment: {(time.time() - exp_loop_start_time) / 60 / len(num_exp_runs)} minutes\n\n", flush=True)

print(f"\n\nFinished all experiments - total runtime: {(time.time() - script_start_time) / 60 / 60} hours", flush=True)