import urllib.request, json 
import pandas as pd
import numpy as np
from copy import deepcopy

# Import dataset
def get_dataset(dataset_name):
    if dataset_name == "fake_data":
        return pd.DataFrame({
            "absurd_column_name":[1,1,3,4,5],
            "another_absurd_column_name":[1,1,1,1,1],
            "yet_another_col_name":["FN","PS","FN","FN","FN"],
            "data":[1,1,3,4,5],
            "bullshit2":[1,1,1,1,1]
            })
    elif dataset_name == "fake_data2":
        return pd.DataFrame({
            "absurd_column_name":[1,2,3,4,5],
            "another_absurd_column_name":[1,1,1,1,1],
            "yet_another_col_name":["FN","FN","FN","FN","FN"],
            "data":[1,2,3,4,5],
            "bullshit2":[1,1,1,1,1]
            })
    else:
        return pd.DataFrame()

# Check if tags are equal
def are_tags_equal(tag1, tag2):
    if tag1 == tag2:
        return True
    else:
        return False

# Import Metadata
def get_metadata():
    with open("/Users/julien/QUORUM/Priorisation2/DataProvider/DataLake2.json") as f:
        metadata = json.load(f)
    return metadata

# Import correlation file
def get_keys_correlation():
    with open("/Users/julien/QUORUM/Priorisation2/DataProvider/Keys.json") as f:
        result = json.load(f)
    return result

# Look for a feature with a specified tag
def get_datasets_from_tags(tags:dict, metadata:dict):

    tags = deepcopy(tags)
    metadata = deepcopy(metadata)
    datasets = {}
    metadata_datasets = {}
    feature_col_names = {}

    for dataset_name in metadata:
        for feature_name in metadata[dataset_name]["raw_data"]:
            if metadata[dataset_name]["raw_data"][feature_name]["tags"] == tags:
                dataset = get_dataset(dataset_name)
                datasets[dataset_name] = dataset
                metadata_datasets[dataset_name] = metadata[dataset_name]
                feature_col_names[dataset_name] = feature_name
            else:
                # Should be able to see if tags is convertible, and convert it if possible
                pass
    return datasets, metadata_datasets, feature_col_names

# Look for dataset with a specific set of keys
def get_dataset_from_key_code(key_codes:dict, metadata:dict):

    key_codes = deepcopy(key_codes)
    metadata = deepcopy(metadata)
    new_metadata = {}
    datasets = {}
    for dataset_name in metadata.keys():
        if set(metadata[dataset_name]["keys"].keys()) == set(key_codes):
            new_metadata[dataset_name] = metadata[dataset_name]
            datasets[dataset_name] = get_dataset(dataset_name)
    return datasets, new_metadata



# Convert dataset to a specific set of keys
def convert_dataset_to_key_code_version(key_codes:dict, datasets:dict, metadata:dict):
    result = {}
    keys_correlation = get_keys_correlation()
    key_codes = deepcopy(key_codes)
    datasets = deepcopy(datasets)
    metadata = deepcopy(metadata)
    new_metadata = {}

    for dataset_name, dataset in datasets.items():
        # Change key version if needed
        if set(metadata[dataset_name]["keys"].keys()) == set(key_codes):
            new_metadata[dataset_name] = metadata[dataset_name]

            for key in metadata[dataset_name]["keys"]:

                # previous_columns stores the columns to be changed in dataset for a specific key
                previous_columns = {}
                for key_element, col_name in metadata[dataset_name]["keys"][key].items():
                    previous_columns[key_element] = dataset[col_name]
                previous_columns = pd.DataFrame(previous_columns)
                

                # old_version stores the version of key in dataset as in keys_correlation
                for version in keys_correlation[key]["versions"]:
                    if set(version.keys()) == set(previous_columns.keys()):
                        old_version = pd.DataFrame(version)
                
                # new_version stores the version of key in key_codes as in keys_correlation
                for version in keys_correlation[key]["versions"]:
                    if set(version.keys()) == set(key_codes[key].keys()):
                        new_version = pd.DataFrame(version)
                        
                
                # new_columns stores the right version of the key in dataset
                new_columns = pd.DataFrame(columns=new_version.columns)
                for i in range(len(previous_columns)):
                    j = 0
                    while (not old_version.loc[[j]].sort_index(axis=1).equals(previous_columns.loc[[i]].sort_index(axis=1))) and j < len(old_version)-1:
                        j = j + 1
                    if (j == len(old_version)-1) and (not old_version.loc[[j]].sort_index(axis=1).equals(previous_columns.loc[[i]].sort_index(axis=1))) :
                        empty_row = pd.Series([np.NaN for i in range(len(new_columns.columns))], index = new_columns.columns)
                        new_columns = new_columns.append(empty_row, ignore_index = True)
                    else:
                        new_columns = new_columns.append(new_version.loc[j], ignore_index=True)

                # Delete previous version of key, add new version, and update metadata
                for key_element, col_name in metadata[dataset_name]["keys"][key].items():
                    dataset = dataset.drop(col_name, axis = 1)
                dataset = pd.concat([dataset,new_columns],axis = 1, join_axes = [dataset.index])
                new_metadata[dataset_name]["keys"][key] = {}
                for key_element in new_version:
                    new_metadata[dataset_name]["keys"][key][key_element] = key_element

            result[dataset_name] = dataset           
                

    return (result,new_metadata)

# Returns how muck keys there are in the datasets
def filter_dataset(datasets:dict, keys:dict, metadata:dict, compliteness_ratio:float):

    filtered_datasets = {}
    compliteness_datasets = {}
    for dataset_name, dataset in datasets.items():

        # Fill result with elements from dataset present in keys
        filtered_dataset = pd.DataFrame(columns=dataset.columns)
        length = len(next(iter(next(iter(keys.values())).values())))
        all_keys = {}

        for i in range(length):
            is_in_keys = True
            for key in keys.keys():
                for key_element in keys[key].keys():
                    col_name = metadata[dataset_name]["keys"][key][key_element]
                    all_keys[col_name] = keys[key][key_element]
                    if dataset[col_name].loc[i] not in keys[key][key_element]:
                        is_in_keys = False
            if is_in_keys:
                filtered_dataset = filtered_dataset.append(dataset.loc[i])
        
        compliteness = len(filtered_dataset)/length
        if compliteness > compliteness_ratio:
            filtered_datasets[dataset_name] = filtered_dataset
            compliteness_datasets[dataset_name] = compliteness

    return filtered_datasets,compliteness_datasets

# Look for features (with the same tags in local and training datasets)
def lookup_features(local_metadata:dict, local_datatsets:dict, training_metadata:dict, training_datasets:dict, compliteness:dict):

    local_metadata = deepcopy(local_metadata)
    training_metadata = deepcopy(training_metadata)
    new_local_datasets = deepcopy(local_datatsets)
    new_training_datasets =  deepcopy(training_datasets)
    new_training_metadata = {}
    new_local_metadata = {}
    new_compliteness = {}

    for training_dataset_name in training_metadata:
        for training_feature, training_feature_metadata in training_metadata[training_dataset_name]["raw_data"].items():
            for local_dataset_name in local_metadata:
                for local_feature, local_feature_metadata in local_metadata[local_dataset_name]["raw_data"].items():
                    if are_tags_equal(local_feature_metadata["tags"], training_feature_metadata["tags"]):
                        if local_dataset_name not in new_local_metadata:
                            new_local_metadata[local_dataset_name] = deepcopy(local_metadata[local_dataset_name])
                            new_local_metadata[local_dataset_name]["raw_data"] = {}
                        new_local_metadata[local_dataset_name]["raw_data"][local_feature] = local_metadata[local_dataset_name]["raw_data"][local_feature]
                        if training_dataset_name not in new_training_metadata:
                            new_training_metadata[training_dataset_name] = deepcopy(training_metadata[training_dataset_name])
                            new_training_metadata[training_dataset_name]["raw_data"] = {}
                        new_training_metadata[training_dataset_name]["raw_data"][training_feature] = training_metadata[training_dataset_name]["raw_data"][training_feature]
    

    # Selecting columns in local_datasets
    for local_dataset_name in new_local_metadata: 
        new_columns = []
        for key in local_metadata[local_dataset_name]["keys"].values():
            for column_name in key.values():
                new_columns.append(column_name)
        for column_name in local_metadata[local_dataset_name]["raw_data"].keys():
            new_columns.append(column_name)
        new_local_datasets[local_dataset_name] = local_datatsets[local_dataset_name][new_columns]
        new_compliteness[local_dataset_name] = compliteness[local_dataset_name]

    # Selecting columns in training_datasets
    for training_dataset_name in new_training_metadata:
        new_columns = []
        for key in training_metadata[training_dataset_name]["keys"].values():
            for column_name in key.values():
                new_columns.append(column_name)
        for column_name in training_metadata[training_dataset_name]["raw_data"].keys():
            new_columns.append(column_name)
        new_training_datasets[training_dataset_name] = training_datasets[training_dataset_name][new_columns]

    return new_local_metadata, new_local_datasets, new_training_metadata, new_training_datasets, new_compliteness


# Big boom function
def request(var_tag:dict, local_key_code:dict, compliteness_ratio:float):

    metadata = get_metadata()

    '''
    Look for datasets with right tag
        -> var_datasets stores all datasets with the right tag
        -> var_metadata stores their metadata
        -> var_col_names stores the right col_name
    '''
    var_datasets,var_metadata,var_col_names = get_datasets_from_tags(var_tag, metadata)

    '''
    Select datasets with the right local_key and right version
        -> local_datasets stores datasets with the right key and with the right version
        -> local_metadata stores their metadata
    '''
    local_datasets,local_metadata = convert_dataset_to_key_code_version(local_key_code,var_datasets,var_metadata)

    '''
    Filter to local_keys
        -> filtered_local_datasets stores the filtered local datasets
        -> compliteness stores their compliteness_ratio
    '''
    filtered_local_datasets,compliteness = filter_dataset(local_datasets,local_key_code,local_metadata,compliteness_ratio)

    '''
    Check compliteness and return variable with keys
    '''
    result = {}
    if compliteness == {}:
        return "Need to infer"
    else:
        for dataset_name, dataset in filtered_local_datasets.items():

            key_col_names = []
            for key in metadata[dataset_name]["keys"].values():
                key_col_names += [col_name for col_name in key.values()]             

            result[dataset_name] = dataset[[var_col_names[dataset_name]]+key_col_names]
    return result, compliteness

def get_features(var_datasets:list, local_key_code:dict, compliteness_ratio:float):
    
    metadata = get_metadata()

    '''
    Look for local_datasets, convert it to local_key_code, and check completeness
    -> local_datasets stores datasets with the right key
    -> local_metadata stores their metadata
    '''
    local_datasets,local_metadata = get_dataset_from_key_code(local_key_code,metadata)
    local_datasets,local_metadata = convert_dataset_to_key_code_version(local_key_code,local_datasets,local_metadata)
    filtered_local_datasets,compliteness = filter_dataset(local_datasets,local_key_code,local_metadata,compliteness_ratio)
    
    '''
    Look for training_datasets
    -> training_datasets stores all possible datasets with the same keys as var_datasets => Possible training_features
    -> training_metadata stores their metadata
    '''
    filtered_training_datasets = {}
    training_metadata = {}
    for var_dataset_name in var_datasets:
        var_key_code = metadata[var_dataset_name]["keys"] 
        filtered_training_datasets[var_dataset_name], training_metadata[var_dataset_name] = get_dataset_from_key_code(var_key_code,metadata)


    '''
    Look for features
    '''

    local_features_metadata = {}
    local_features_datasets = {}
    training_features_metadata = {}
    training_features_datasets = {}
    compliteness_var = {}
    for var_dataset_name in var_datasets:
        local_features_metadata[var_dataset_name], local_features_datasets[var_dataset_name], training_features_metadata[var_dataset_name], training_features_datasets[var_dataset_name], compliteness_var[var_dataset_name] = \
        lookup_features(local_metadata,filtered_local_datasets,training_metadata[var_dataset_name],filtered_training_datasets[var_dataset_name],compliteness)

    return local_features_metadata, local_features_datasets, training_features_metadata, training_features_datasets, compliteness_var



# Tests

var_datasets = ["fake_data"]
local_key_code = {
    "polling_station":{
        "municipality_code_insee":[11,11,7],
        "BDV_code":[1,1,3]
        },
    "candidate":{
        "candidate_id":[1,1,3],
        "nuance_initials":["FN","PS","FF"]
        },
    "bullshit":{
        "bullshit1":[1,1,1]
        }
    }
compliteness_ratio = 0.2

(a,b,c,d,e) = get_features(var_datasets,local_key_code,compliteness_ratio)
print("local_features_metadata: ","\n",a,"\n")
print("local_features_datasets: ","\n",b,"\n")
print("training_features_metadata: ","\n",c,"\n")
print("training_features_datasets: ","\n",d,"\n")
print("compliteness_var: ","\n",e,"\n")
