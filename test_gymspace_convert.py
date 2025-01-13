from libs.TrussFrameASAP.gymenv.cantileverenv_convert_gymspaces import ObservationBijectiveMapping, ActionBijectiveMapping, ObservationDownSamplingMapping
from tqdm import tqdm  # Import tqdm for progress bar
import numpy as np
import time

def test_observation_downsampling_mapping():
    """
    Test the downsampling mapping to ensure all observation tuple values encode to unique values.
    """
    # Example parameters for testing
    # real env param
    # framegrid_size_x = 10  # Original frame grid x-dimension
    # framegrid_size_y = 5  # Original frame grid y-dimension
    # inventory_array = [8, 3]  # Inventory capacities for each frame type
    # kernel_size = 2
    # stride = 1
    
    framegrid_size_x = 4  # Original frame grid x-dimension
    framegrid_size_y = 4  # Original frame grid y-dimension
    inventory_array = [4, 2]  # Inventory capacities for each frame type
    kernel_size = 3
    stride = 2

    # Instantiate the mapping
    mapping = ObservationDownSamplingMapping(framegrid_size_x, framegrid_size_y, inventory_array, kernel_size, stride)

    # Store encoded values to ensure uniqueness
    encoded_observations = set()

    # Calculate the bounds for downsampled grid
    downsampled_x = (framegrid_size_x + kernel_size - 1) // kernel_size
    downsampled_y = (framegrid_size_y + kernel_size - 1) // kernel_size

    # Iterate over all possible values in the downsampled grid and inventory
    num_grid_values = mapping.grid_encoding_base
    print(f'num_grid_values: {num_grid_values}')
    # TODO get range of grid values
    num_grid_values_range = range(-1, num_grid_values - 1)
    num_inventory_values = [cap + 1 for cap in inventory_array]

    from itertools import product

    # Calculate the total number of combinations
    total_inventory_combinations = np.prod([cap + 1 for cap in inventory_array])
    total_grid_combinations = (num_grid_values) ** (framegrid_size_x * framegrid_size_y)
    total_combinations = total_grid_combinations * total_inventory_combinations
    print(f"Total combinations to test: {total_combinations}")

    # Start timing
    start_time = time.time()

    # Add the progress bar
    with tqdm(total=total_combinations, desc="Processing observations") as pbar:

        # Generate all possible downsampled grids
        for grid_values in product(num_grid_values_range, repeat=(framegrid_size_x * framegrid_size_y)):
            org_frame_grid = np.array(grid_values).reshape(framegrid_size_x, framegrid_size_y)

            # Generate all possible inventory combinations
            for inventory_values in product(*[range(cap + 1) for cap in inventory_array]):
                inventory = np.array(inventory_values)

                # Encode the observation
                encoded_value = mapping.encode(org_frame_grid, inventory)

                # Check for duplicates
                observation_tuple = (tuple(map(tuple, org_frame_grid)), tuple(inventory))  # Hashable form
                if encoded_value in encoded_observations:
                    print(f"Duplicate found for encoded value: {encoded_value} from observation: {observation_tuple}")
                    return False
                encoded_observations.add(encoded_value)
                pbar.update(1)

        print("All tests passed. The mapping is injective!")
        # End timing
        end_time = time.time()

        # Calculate and print elapsed time
        elapsed_time = end_time - start_time
        print(f"Total time taken: {elapsed_time:.2f} seconds")
        return True

def test_observation_bijective_mapping():
    """
    Test the bijective mapping to ensure all integers in the total range decode to unique values.
    """
    # Example parameters for the mapping
    frame_grid_size_x = 3  # Smaller size for quicker testing
    frame_grid_size_y = 3
    inventory_array = [2, 3]  # Example inventory capacities

    # Create the mapping
    mapping = ObservationBijectiveMapping(frame_grid_size_x, frame_grid_size_y, inventory_array)

    # Store decoded observations to ensure uniqueness
    decoded_observations = set()

    # Iterate over all possible encoded integers
    print(f'Total observation space size: {mapping.total_space_size}')
    # for encoded_value in range(mapping.total_space_size):
    for encoded_value in tqdm(range(mapping.total_space_size), desc="Testing Observations"):
        # Decode the integer into an observation
        decoded_observation = mapping.decode(encoded_value)

        # Convert `frame_grid` and `inventory` to hashable types
        frame_grid_tuple = tuple(map(tuple, decoded_observation[0]))  # Convert 2D array to nested tuples
        inventory_tuple = tuple(decoded_observation[1])  # Convert 1D array to tuple
        hashable_observation = (frame_grid_tuple, inventory_tuple)

        # Check for duplicates
        if hashable_observation in decoded_observations:
            print(f"Duplicate found: {hashable_observation} for encoded value {encoded_value}")
            return False
        decoded_observations.add(hashable_observation)

        # Encode back to ensure consistency
        # print(f'decoded_observation: {decoded_observation}')
        frame_grid, inventory = decoded_observation
        re_encoded_value = mapping.encode(frame_grid, inventory)
        if re_encoded_value != encoded_value:
            print(f"Inconsistent mapping: {decoded_observation} encodes to {re_encoded_value} instead of {encoded_value}")
            return False

    print("All tests passed. The mapping is bijective and exhaustive!")

    return True


def test_action_bijective_mapping():
    """
    Test the bijective mapping for actions to ensure all integers in the total range decode to unique values.
    """
    # Example parameters for the action mapping
    frame_grid_size_x = 5
    frame_grid_size_y = 5
    freeframe_min = 1
    freeframe_max = 3  # Example bounds for freeframe types

    # Create the mapping
    mapping = ActionBijectiveMapping(frame_grid_size_x, frame_grid_size_y, freeframe_min, freeframe_max)

    # Store decoded actions to ensure uniqueness
    decoded_actions = set()

    # Iterate over all possible encoded integers
    for encoded_value in range(mapping.total_space_size):
        # Decode the integer into an action
        decoded_action = tuple(mapping.decode(encoded_value))

        # Check for duplicates
        if decoded_action in decoded_actions:
            print(f"Duplicate found: {decoded_action} for encoded value {encoded_value}")
            return False
        decoded_actions.add(decoded_action)

        # Encode back to ensure consistency
        re_encoded_value = mapping.encode(list(decoded_action))
        if re_encoded_value != encoded_value:
            print(f"Inconsistent mapping: {decoded_action} encodes to {re_encoded_value} instead of {encoded_value}")
            return False

    print("All tests passed. The action mapping is bijective!")
    return True

# Run the test
if __name__ == "__main__":
    # test_observation_bijective_mapping()
    # test_action_bijective_mapping()
    test_observation_downsampling_mapping()