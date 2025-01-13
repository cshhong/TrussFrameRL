'''
FINAL : load300_n3000 !!
[TODO edit TrussFrameASAP directory path]
'''
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.utils.save_video import save_video
import imageio
import libs.TrussFrameASAP.gymenv  # Register the custom environments with __init__.py
from tqdm import tqdm  # Import tqdm for the progress bar

import os
import h5py

from libs.TrussFrameASAP.PerformanceMap.h5_utils import *
from libs.TrussFrameASAP.PerformanceMap.render_loaded import RenderLoaded
from libs.TrussFrameASAP.PerformanceMap.perfmap import *
from libs.TrussFrameASAP.PerformanceMap.get_img_url import scrape_image_links

from sklearn.cluster import DBSCAN # for clustering


def video_save_trigger(n_epi):
    # if n_epi % VIDEO_INTERVAL == 0:
    #     print("Saving render!")
    #     return True 
    # else:
    #     return False
    return True

def random_rollout(env, h5f, num_eps_start=0, num_episodes=25, steps_per_episode=1000, video_render_dir=None):
    """
    Run episodes with random actions and save results to an open HDF5 file.

    Parameters:
    - h5f (h5py.File): An open HDF5 file object where episode data will be saved.
    - num_eps_start (int): The starting index for episode numbering. Defaults to 0.
    - num_episodes (int): The number of episodes to run. Defaults to 25.
    - steps_per_episode (int): The maximum number of steps per episode. Defaults to 1000.
    - render_dir (str): Directory path for saving any rendering outputs. Defaults to "./".

    Output:
    - Modifies the open HDF5 file with saved episode data.

    # Returns:
    # - list that contains: each episode instance[term_eps_ind, max_disp, [(v1, v2). ...] , [frame1.type_structure.idx, ] ]
    #   - terminated episode index (int)
    #   - maximum deflection observed during the episode (float)
    #   - list of failed elements, where each failed element is represented as a tuple of vertex IDs (list of tuples)
    #   - frame_grid (np.array)
    """
    term_eps_idx = 0 # only store terminated episodes
    # Run episodes with random actions
    for episode in range(num_eps_start, num_eps_start+num_episodes):
        # add progress line
        obs = env.reset()
        
        for step in tqdm(range(steps_per_episode), desc=f"Episode {episode+1} / {num_episodes}"):
            action = env.action_space.sample()  # Random action int
            obs, reward, terminated, truncated, info = env.step(action)
            print(f'obs : {obs} | step {step} : {action} | reward : {reward} | terminated : {terminated} | truncated : {truncated}')
            if truncated:
                # print("Truncated!")
                break
            
            # If render mode is rgb_list, save video at 
            # episode ended within inventory limit
            if terminated:
                if env.render_mode == "rgb_list":
                    assert video_render_dir is not None, "Please provide a directory path for saving the rendered video."
                    save_video(
                                frames=env.get_render_list(),
                                video_folder=video_render_dir,
                                fps=env.metadata["render_fps"],
                                # video_length = ,
                                # name_prefix = f"Episode ",
                                episode_index = episode+1,
                                # step_starting_index=step_starting_index,
                                episode_trigger = video_save_trigger 
                    )

                # print(f"max deflection : {env.unwrapped.max_deflection}")
                # Save data for episodes only at termination to the HDF5 file
                # print(f"Saving episode {term_eps_idx} to HDF5 file.")
                save_episode_hdf5(h5f, term_eps_idx, env.unwrapped.curr_fea_graph, env.unwrapped.frames, env.unwrapped.curr_frame_grid)
                # save selective data to list 
                
                term_eps_idx += 1

                # Flush data to disk (optional - may slow down training)
                h5f.flush()

                break
        
        # Display the frames for this episode
        # display_frames_as_gif(frames, episode)

    # Close the environment
    env.close()

if __name__ == "__main__":

    name = 'load300_n3000_binary'
    #TODO direct to h3 directory!! h5dir = 'TrussFrameASAP/PerformanceMap/h5files'
    save_hdf5_filename = os.path.join(h5dir, f'{name}.h5')
    render_mode = "debug_all"

    # # # # # Number of episodes and steps per episode
    total_episodes = 60 # about 30% terminate (connected structures)
    steps_per_episode = 300 # considering invlaid steps, enough to create terminated episodes with high probability 
    
    # ################ Set Env ################    
    # if render_mode == "rgb_list":
    #     render_dir = os.path.join("./TrussFrameASAP/videos/", name) # render_mode == "rgb_list"
    # elif render_mode == "rgb_end":
    #     render_dir = os.path.join("TrussFrameASAP/PerformanceMap/render_perf/",name) # render_mode != "rgb_end"
    # else:
    #     render_dir = os.path.join("TrussFrameASAP/PerformanceMap/render_perf/",name) # not used but required
    # env = gym.make("Cantilever-v0", render_mode=render_mode, render_dir=render_dir)

    # ################ Save HDF5 File ################    
    # Open the HDF5 file at the beginning of rollout
    # h5f = h5py.File(save_hdf5_filename, 'a', track_order=True)  # Use 'w' to overwrite or 'a' to append
    # store data in creation order (default is alphabetical order)
    
    ################ Perform Rollout ################    
    # try: # Ensure that the HDF5 file is properly closed, even if an exception occurs during training.
    #     #Training loop and data saving
    #     random_rollout(env, 
    #                 h5f, 
    #                 num_eps_start = 0, # make sure this is set right!
    #                 num_episodes=total_episodes, 
    #                 steps_per_episode=steps_per_episode,
    #                 video_render_dir=render_dir,
    #                 )

    #     # log env properties needed for rendering loaded data 
    #     save_env_render_properties(h5f, env)
    # finally:
    #     h5f.close()
    
    ################ Load HDF5 File ################
    load_hdf5_filename = save_hdf5_filename
    # print_hdf5_structure(load_hdf5_filename)
    
    ################ Mapping frame grids to 2D UMAP ################
    stacked_frame_grids = load_framegrids_hdf5(load_hdf5_filename)
    stacked_frame_grids_binary = np.where(stacked_frame_grids > 0, 1, 0)
    MAX_DISPLACEMENTS = load_max_deflection_hdf5(load_hdf5_filename)
    FAILED_ELEMENTS = load_failed_elements_hdf5(load_hdf5_filename)
    ALLOWABLE_DISPLACEMENT = load_allowable_deflection_hdf5(load_hdf5_filename)

    idx_with_failed_elements = [i for i, failed in enumerate(FAILED_ELEMENTS) if len(failed) > 0]
    print(f'Number of failed designs : {len(idx_with_failed_elements)} / {len(MAX_DISPLACEMENTS)}')

    # ################ UMAP dimensionality reduction on framegrids ################ 
    # umap_n_neighbors = 5 # low n_neighbor preserves local, high n_neighbor preserves global structure, default is 15
    # # umap_2d = umap_reduce_framegrid(stacked_frame_grids, n_neighbors=umap_n_neighbors, min_dist=0.0005) # in order of idx value
    # umap_1d = umap_reduce_framegrid_1D(stacked_frame_grids, n_neighbors=umap_n_neighbors, min_dist=0.0005) # in order of idx value
    
    # # # ################ Clustering points for selective rendering (finetune) ################
    # # get clusters with DBSCAN
    # dbscan_num_core = 4 # high n_core creates strict requirement for points to be defined as cluster default value is dim*2 =4
    # dbscan_eps = 0.5 # The maximum distance between two samples for one to be considered as in the neighborhood of the other. default value is 0.5
    # dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_num_core)
    # cluster_labels = dbscan.fit_predict(umap_1d)
    # # num_clusters = len(set(cluster_labels)) - 1 # number of clusters excluding noise points
    # num_clusters = len(set(cluster_labels)) - 1 if -1 in cluster_labels else len(set(cluster_labels))
    # print(f'UMAP n_neighbors : {umap_n_neighbors} | DBSCAN n_core : {dbscan_num_core} | DBSCAN eps : {dbscan_eps}| Number of Clusters : {num_clusters}') # number of clusters excluding noise points
    # # ! Check if number of clusters and cluster size is reasonable ! # 
    # # get cluster idx - number of instances in each cluster
    # cluster_idx, cluster_counts = np.unique(cluster_labels, return_counts=True)
    # # print cluster idx - number of instances in each cluster in inline for loop
    # print(f'cluster {cluster_idx} \n {cluster_counts}')

    # # # # # ################# Save cluster information in HDF5 file ################
    # save_cluster_data(load_hdf5_filename, umap_n_neighbors, dbscan_num_core, dbscan_eps, num_clusters, cluster_labels)
    # # save_umap2d_hdf5(load_hdf5_filename, umap_2d)
    # save_umap2d_hdf5(load_hdf5_filename, umap_1d)

    # # # ################# Load cluster information in HDF5 file ################
    # load cluster labels from HDF5 file
    print(f'Loading cluster labels from {load_hdf5_filename}')
    # cluster_labels = load_cluster_labels_hdf5(load_hdf5_filename)
    CLUSTER_LABELS = load_cluster_labels_hdf5(load_hdf5_filename)
    UMAP2D = load_umap2d_hdf5(load_hdf5_filename)
    UMAP1D = UMAP2D[:,0]
    # umap_2d = load_umap2d_hdf5(load_hdf5_filename)
    # Only rendering episodes from clusters with low, median, and high max displacement
    SELECTED_CLUSTER_IDX = select_from_clusters(MAX_DISPLACEMENTS, CLUSTER_LABELS, n_select=3)

    # # ################ Query images for selective episodes ################
    alt_name = 'load300_n3000'
    # render_dir = os.path.join("TrussFrameASAP/PerformanceMap/render_perf/", alt_name) 
    # # load environment render properties
    # render_properties = load_env_render_properties(load_hdf5_filename)
    # # print(f'Render Properties : \n {render_properties}')
    # render_loader = RenderLoaded(render_properties) # create RenderLoaded object to selectively load and render episodes
    
    # for cluster in SELECTED_CLUSTER_IDX:
    #     for q_eps_idx in cluster:
    #         # Render One Episode
    #         # load one episode of data from an HDF5 file.
    #         loaded_fea_graph, loaded_frames, loaded_frame_grid = load_episode_hdf5(load_hdf5_filename, q_eps_idx)
    #         ## Render one episode from hdf5 file 
    #         file_dir = os.path.join(render_dir, f"end_{q_eps_idx}")
    #         render_loader.render_loaded(file_dir, loaded_fea_graph, loaded_frames)

    # # # ################ Create Cluster image plot ################
    # create_cluster_image_plot(SELECTED_CLUSTER_IDX, render_dir, name)
    
    ################ Create csv with idx, image url ################
    # 3D performance map : 2D UMAP of shapes (framegrid) + z axis max deformation 
    idx_imgurl_dir = 'TrussFrameASAP/PerformanceMap/idx_imgurl_csv'
    IMG_CSV_NAME = os.path.join(idx_imgurl_dir, f'{name}_imgurls.csv') # csv file with idx and image urls column

    # create csv file with idx and img_url columns **This will wipe the existing url values!**
    # init_csv_from_hdf5(load_hdf5_filename, IMG_CSV_NAME) 
    
    ################ Populate csv file with image URLs from Dropbox ################
    ##### !! Have to have moved rendered images to Dropbox folder !! #####
    # Populate the 'img_url' column in the CSV file with direct image links from a Dropbox folder.
    # Dropbox API Access Token
    # API Console : https://www.dropbox.com/developers/apps/info/a44uvsxdkczevsi

    # DROPBOX_FOLDER = f"/{alt_name}" # relative path within Dropbox/Apps/TrussFrame/
    # ACCESS_TOKEN = "sl.u.AFZ-B9w_WwPf2KFPz_DoSYUHuwB6YRGbEW4P9cgfApGNnrGcxoMFh8TlyDWE9UW4-2ELWRfh2OSCTe3jMm5IchW5ttB1IPCYHGKTwQWysMMaXVs_jBRBUMLsSslrs9kuxUrLGrK_I7O1gQrUnsMg8dQ-WkOJA1pz5xLFrPtAw1QWvZnufitS5JxUCCqvImsnpyGPF9TMEtgz1IOP2c-X7M9sBWjTyw4jG-3lEF3FCuS4PrHxHrcodwr5Sig7exVqy-S_5najSu-rKVphYk0sMY4Vx_YGoqJoKzxDdaWD7LHDPNbiDWOborWS1-3T9jEOVOKyQfIpfhpb1JEflScI16aQw4X4O2WZa9OSP5fgjQwuLrCacg_jETQZzt9aVSfjedm79wPSbNdGOmvm52Pcq5HaEQ_L_FJM38lq--xjs4ofLzuX684wWsRQwJhF25yPDkZHMhd8ckA5kvJCoh7OwW4RMe9HXZWr8ZcTSa4Bd9uS7nQJpra-eb8ryKiUpRQHsDBLWriHC2x-HsjX5SH71C3K5S81izJzwB0xmt8uoBMNF-MVzy0BckXR3s8DEGtUSCJ7G_6-OTQgdhw2YFZg7wwjQRCeR5fkgU9oKzYH1XwpvFuGJJvwVy8T6ymQePUmCrvJ3jiuTfprtpzytiY8G83pFzLNo57zLOCmyrXHPC-4ffVqNRwmaaxrbjmCUfhaD89iDoDtyvO9byTaenu6UN2-dRIfWIVlnx_ufCs5bKa-hDjD4WaHQUa8qSdlobZLMIDWOfoYxBfP-gg64bhRN5k90-Ry5lSXC7xm-nUchfYRdUwbnAgfZPpF8a8NEbnjqsv5sGq2l4dP86QvCbGn8JvgEAyy6s9W_PBeenRQ5lOQOzJT0ZLdVO4Pft2W1JG3br4-wNDTko4DMKq4GWIu2_x807AqP1hrKrqQDW06RMYIP3Nm3AdcWvVJYMpzuOSQ4AY6g7e8kmMoD3KvWPK5LDLPKgYLhEwUCJXUucn8ZdegbbuUrEHEZqByXL6Ncd48WR8PoNJKbAfoQk9eS2fYKKjLwVcy3-L9AYKzsk_eu4_2yaTnO4jsblgSMBvgdm4FYzXq8DS5vLSlDFvUNmbo1Hdo5P85M8keBxVAIx22e_FVMh_OZ9cDUK_p8PAHnf11hK_OiAhrvaBpGEm2CZ8UxBAMZDFnhh33AYTU4OSMY-hb-hrWM3sPZ3vUEYE8I-bn5aPk-yWqQtx4DA-fYGiJxaJ9ErWQIT1pVZYV9AgkzSLRQQJmKxsxi-p8Cy4o4TKz6eNqKGd_WEK_eJLIAZnSE4owouiL5sGmaBfbokDqS4Cs1g"
    # scrape_image_links(ACCESS_TOKEN, DROPBOX_FOLDER, IMG_CSV_NAME)
    
    # ################ Create Interactive Map ################
    # create_interactive_performance_map_cluster( 
    #                                     max_displacements=MAX_DISPLACEMENTS, 
    #                                     all_failed_elements=FAILED_ELEMENTS, 
    #                                     allowable_displacement=ALLOWABLE_DISPLACEMENT, 
    #                                     img_url_csv=IMG_CSV_NAME, 
    #                                     umap_2d=UMAP2D,
    #                                     cluster_labels = CLUSTER_LABELS,
    #                                     selected_cluster_idx = SELECTED_CLUSTER_IDX,
    #                                     gamma=20, 
    #                                     num_z_ticks=6,
    #                                     marker_size=5,
    #                                     )
    
    create_interactive_performance_map_2D(
                                        max_displacements=MAX_DISPLACEMENTS,
                                        all_failed_elements=FAILED_ELEMENTS,
                                        allowable_displacement=ALLOWABLE_DISPLACEMENT,
                                        img_url_csv=IMG_CSV_NAME,
                                        umap_1d=UMAP1D,
                                        cluster_labels=CLUSTER_LABELS,
                                        selected_cluster_idx=SELECTED_CLUSTER_IDX,
                                        gamma=20,
                                        marker_size=5,
                                    )

