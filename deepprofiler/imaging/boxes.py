import os

import numpy as np
import pandas as pd

X_KEY = "Nuclei_Location_Center_X"
Y_KEY = "Nuclei_Location_Center_Y"

#################################################
## BOUNDING BOX HANDLING
#################################################

def get_locations(image_key, config, random_sample=None, seed=None):
    if config["dataset"]["locations"]["mode"] == "single_cells":
        return get_single_cell_locations(image_key, config, random_sample, seed)
    elif config["dataset"]["locations"]["mode"] == "full_image":
        return get_full_image_locations(image_key, config, random_sample, seed)
    else:
        return None

#################################################
# Get single cell locations from the CSV files
#################################################

def get_single_cell_locations(image_key, locations_path, random_sample=None, seed=None):
    # CSV files are expected to be stored in this format: plate/well-site-Nuclei.csv
    keys = image_key.split("/")
    locations_file = "{}/{}-{}.csv".format(
        keys[0],
        keys[1],
        "Nuclei"
    )
    print(image_key, locations_path)
    # Identify the location of the file
    #locations_path = os.path.join(config["paths"]["locations"], locations_file)
    locations_path = os.path.join(locations_path, locations_file)
    if os.path.exists(locations_path):
        # If the file exists sample a few cells or return all of them
        locations = pd.read_csv(locations_path)

        if random_sample is not None and random_sample < len(locations):
            locations.sample(random_sample, random_state=seed)
            locations.values.astype(float).tolist()
            for i in range(len(locations)):
                locations[i] = [locations[i][0] - 48, locations[i][1] - 48, locations[i][0] + 48, locations[i][1] + 48]
            return locations
        else:
            locations.values.astype(float).tolist()
            for i in range(len(locations)):
                locations[i] = [locations[i][0] - 48, locations[i][1] - 48, locations[i][0] + 48, locations[i][1] + 48]
            return locations
    else:
        # If the file does not exist return an empty dataframe
        return pd.DataFrame(columns=[X_KEY, Y_KEY])


#################################################
# Get full image regions that cover a large area
#################################################

def get_full_image_locations(image_key, config, random_sample, seed):
    cols = config["dataset"]["images"]["width"]
    rows = config["dataset"]["images"]["height"]
    coverage = config["dataset"]["locations"]["area_coverage"]
    cols_margin = cols - int(cols * coverage)
    rows_margin = rows - int(rows * coverage)
 
    data = None
    if coverage == 1.0:
        # If the area coverage is all the image use the center of the image
        data = [[cols/2, rows/2]]
    else:
        # Otherwise, generate multiple regions
        if random_sample is not None:
            # Generate random region centers to create random crops for training
            cols_pos = np.random.randint(low=-cols_margin/2, high=cols_margin/2, size=random_sample) + cols/2
            rows_pos = np.random.randint(low=-rows_margin/2, high=rows_margin/2, size=random_sample) + rows/2
            data = [[cols_pos[i], rows_pos[i]] for i in range(random_sample)]
        elif random_sample is None:
            # Generate five regions that cover the corners and center of the image for validation and profiling
            cols_pos = [cols/2 - cols_margin/2, cols/2 - cols_margin/2, cols/2 + cols_margin/2, cols/2 + cols_margin/2, cols/2]
            rows_pos = [rows/2 - rows_margin/2, rows/2 + rows_margin/2, rows/2 - rows_margin/2, rows/2 + rows_margin/2, rows/2]
            data = [[cols_pos[i], rows_pos[i]] for i in range(5)]

    return pd.DataFrame(data=data, columns=[X_KEY, Y_KEY])



#################################################
# Use cell centers to prepare bounding boxes for cropping
#################################################

def prepare_boxes(batch, config):
    if config["dataset"]["locations"]["mode"] == "single_cells":
        # Set the configured box_size to define bounding boxes
        return get_cropping_regions(batch, config, config["dataset"]["locations"]["box_size"])

    elif config["dataset"]["locations"]["mode"] == "full_image":
        # Use the minimum side of the image to define bounding boxes
        side = min(config["dataset"]["images"]["width"], config["dataset"]["images"]["height"])
        coverage = config["dataset"]["locations"]["area_coverage"]
        return get_cropping_regions(batch, config, int(side * coverage))

    else:
        return None

#################################################
# Prepare bounding boxes according to TF crop_and_resize method
#################################################

def get_cropping_regions(batch, config, box_size):
    locations_batch = batch["locations"]
    image_targets = batch["targets"]
    images = batch["images"]
    all_boxes = []
    all_indices = []
    all_targets = [[] for i in range(len(image_targets[0]))]
    all_masks = []
    index = 0

    for locations in locations_batch:
        # Collect and normalize boxes between 0 and 1
        boxes = np.zeros((len(locations), 4), np.float32)
        boxes[:,0] = locations[Y_KEY] - box_size/2
        boxes[:,1] = locations[X_KEY] - box_size/2
        boxes[:,2] = locations[Y_KEY] + box_size/2
        boxes[:,3] = locations[X_KEY] + box_size/2
        boxes[:,[0,2]] /= config["dataset"]["images"]["height"]
        boxes[:,[1,3]] /= config["dataset"]["images"]["width"]
        # Create indicators for this set of boxes, belonging to the same image
        box_ind = index * np.ones((len(locations)), np.int32)
        # Propagate the same labels to all crops
        for i in range(len(image_targets[index])):
            all_targets[i].append(image_targets[index][i] * np.ones((len(locations)), np.int32))
        # Identify object mask for each crop
        masks = np.zeros(len(locations), np.int32)
        if config["dataset"]["locations"]["mask_objects"]:
            i = 0
            for lkey in locations.index:
                y = int(locations.loc[lkey, Y_KEY])
                x = int(locations.loc[lkey, X_KEY])
                patch = images[index][max(y-5,0):y+5, max(x-5,0):x+5, -1]
                if np.size(patch) > 0:
                    masks[i] = int(np.median(patch))
                i += 1
        # Pile up the resulting variables
        all_boxes.append(boxes)
        all_indices.append(box_ind)
        all_masks.append(masks)
        index += 1

    result = (np.concatenate(all_boxes),
              np.concatenate(all_indices),
              [np.concatenate(t) for t in all_targets],
              np.concatenate(all_masks)
             )
    return result
