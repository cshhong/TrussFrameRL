'''
From hdf5 file from training, get renders
'''
import os
import h5py
from TrussFrameASAP.PerformanceMap.h5_utils import *
from TrussFrameASAP.PerformanceMap.render_loaded import RenderLoaded
from TrussFrameASAP.PerformanceMap.perfmap import *
from TrussFrameASAP.PerformanceMap.get_img_url import scrape_image_links
from sklearn.cluster import DBSCAN # for clustering

# make env (temp)
# import gymnasium as gym 
# import TrussFrameASAP.gymenv 

# Get total number of episodes
name = "noinvalidpenalty_disp_inventoryleft_medonly_entcoef0.1"
h5_file_path = f"train_h5/Cantilever-v0_{name}.h5"


results_dir = "training_results"
# Get total number of episodes
with h5py.File(h5_file_path, 'r') as f:
    total_episodes = len(f['/Episodes'])
print(f'Total number of episodes: {total_episodes}')
csv_path = os.path.join(results_dir, name+".csv")
init_csv_from_hdf5(h5_file_path, csv_path)

# # get interactive map of last 3000 episodes
load_hdf5_filename = h5_file_path
eps_range = (total_episodes-501, total_episodes-1)

# ################ Mapping frame grids to 2D UMAP ################
stacked_frame_grids = load_framegrids_hdf5(load_hdf5_filename, eps_range=eps_range)
MAX_DISPLACEMENTS = load_max_deflection_hdf5(load_hdf5_filename, eps_range=eps_range)
FAILED_ELEMENTS = load_failed_elements_hdf5(load_hdf5_filename, eps_range=eps_range)
# ALLOWABLE_DISPLACEMENT = load_allowable_deflection_hdf5(load_hdf5_filename) # TODO f["EnvRenderProperties/allowable_deflection"][()] component not found?
# ALLOWABLE_DISPLACEMENT = 0.075
# env = gym.make("Cantilever-v0", render_mode=None, render_dir="temp")
# with h5py.File(load_hdf5_filename, 'a') as f:
#     save_env_render_properties(f, env)

idx_with_failed_elements = [i for i, failed in enumerate(FAILED_ELEMENTS) if len(failed) > 0]
print(f'Number of failed designs : {len(idx_with_failed_elements)} / {len(MAX_DISPLACEMENTS)}')

############### UMAP dimensionality reduction on framegrids ################ 
umap_n_neighbors = 15 # low n_neighbor preserves local, high n_neighbor preserves global structure, default is 15
umap_2d = umap_reduce_framegrid(stacked_frame_grids, n_neighbors=umap_n_neighbors, min_dist=0.0005) # in order of idx value

# ################ Clustering points for selective rendering (finetune) ################
# get clusters with DBSCAN
dbscan_num_core = 10 # high n_core creates strict requirement for points to be defined as cluster default value is dim*2 =4
dbscan_eps = 0.5 # The maximum distance between two samples for one to be considered as in the neighborhood of the other. default value is 0.5
dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_num_core)
cluster_labels = dbscan.fit_predict(umap_2d)
# num_clusters = len(set(cluster_labels)) - 1 # number of clusters excluding noise points
num_clusters = len(set(cluster_labels)) - 1 if -1 in cluster_labels else len(set(cluster_labels))
print(f'UMAP n_neighbors : {umap_n_neighbors} | DBSCAN n_core : {dbscan_num_core} | DBSCAN eps : {dbscan_eps}| Number of Clusters : {num_clusters}') # number of clusters excluding noise points
# ! Check if number of clusters and cluster size is reasonable ! # 
# get cluster idx - number of instances in each cluster
cluster_idx, cluster_counts = np.unique(cluster_labels, return_counts=True)
# print cluster idx - number of instances in each cluster in inline for loop
print(f'cluster {cluster_idx} \n {cluster_counts}')

 # # ################# Save cluster information in HDF5 file ################
save_cluster_data(load_hdf5_filename, umap_n_neighbors, dbscan_num_core, dbscan_eps, num_clusters, cluster_labels)
save_umap2d_hdf5(load_hdf5_filename, umap_2d)

################ Load cluster information in HDF5 file ################
# load cluster labels from HDF5 file
print(f'Loading cluster labels from {load_hdf5_filename}')
# cluster_labels = load_cluster_labels_hdf5(load_hdf5_filename)
CLUSTER_LABELS = load_cluster_labels_hdf5(load_hdf5_filename)
UMAP2D = load_umap2d_hdf5(load_hdf5_filename)
# Only rendering episodes from clusters with low, median, and high max displacement
SELECTED_CLUSTER_IDX = select_from_clusters(MAX_DISPLACEMENTS, CLUSTER_LABELS, n_select=2)

################ Query images for selective episodes ################
render_dir = os.path.join(results_dir, "selective_renders/"+name) 
# create render_dir if it does not exist
if not os.path.exists(render_dir):
    os.makedirs(render_dir)
# load environment render properties
render_properties = load_env_render_properties(load_hdf5_filename)
# print(f'Render Properties : \n {render_properties}')
render_loader = RenderLoaded(render_properties) # create RenderLoaded object to selectively load and render episodes

for cluster in SELECTED_CLUSTER_IDX:
    for q_eps_idx in cluster:
        org_idx = q_eps_idx + eps_range[0]
        # Render One Episode
        # load one episode of data from an HDF5 file.
        loaded_fea_graph, loaded_frames, loaded_frame_grid = load_episode_hdf5(load_hdf5_filename, org_idx)
        ## Render one episode from hdf5 file 
        file_dir = os.path.join(render_dir, f"end_{org_idx}")
        render_loader.render_loaded(file_dir, loaded_fea_graph, loaded_frames)

# ################ Create Cluster image plot ################
create_cluster_image_plot(SELECTED_CLUSTER_IDX, render_dir, name, eps_start_idx=eps_range[0])
