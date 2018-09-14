import urllib.request, json 
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np

# Import dataset
def get_dataset(dataset_name):
    if dataset_name == "fake_data":
        return pd.DataFrame({
            "absurd_column_name":[1,2,3,4,5],
            "another_absurd_column_name":[1,1,1,1,1],
            "yet_another_col_name":["FN","FN","FN","FN","FN"],
            "data":[1,2,3,4,5]
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



# Look for datatset with a specific set of keys
def convert_dataset_to_key_code(key_codes:dict, datasets:dict, metadata:dict):
    result = {}
    keys_correlation = get_keys_correlation()

    for dataset_name, dataset in datasets.items():
        # Check if dataset has the same key_code
        is_in_key_code = True
        for key in metadata[dataset_name]["keys"].keys():
            if key not in key_codes:
                is_in_key_code = False

        # Change key version if needed
        if is_in_key_code:
            for key in metadata[dataset_name]["keys"]:

                # previous_columns stores the columns to be changed in dataset for a specific key
                previous_columns = {}
                for key_element, col_name in metadata[dataset_name]["keys"][key].items():
                    previous_columns[key_element] = dataset[col_name]
                previous_columns = pd.DataFrame(previous_columns)
                

                # old_version stores the version of key in dataset as in keys_correlation
                for version in keys_correlation[key]["versions"]:
                    is_old_version = True
                    for key_element in version:
                        if key_element not in previous_columns:
                            is_old_version = False
                    if is_old_version == True:
                        old_version = pd.DataFrame(version)
                
                # new_version stores the version of key in key_codes as in keys_correlation
                for version in keys_correlation[key]["versions"]:
                    is_new_version = True
                    for key_element in version:
                        if key_element not in key_codes[key].keys():
                            is_new_version = False
                    if is_new_version == True:
                        new_version = pd.DataFrame(version)
                        
                
                # new_columns stores the right version of they key in dataset
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
                metadata[dataset_name]["keys"][key] = {}
                for key_element in new_version:
                    metadata[dataset_name]["keys"][key][key_element] = key_element

            result[dataset_name] = dataset           
                

    return (result,metadata)

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
    local_datasets,local_metadata = convert_dataset_to_key_code(local_key_code,var_datasets,var_metadata)

    '''
    Filter to local_keys
        -> filtered_local_datasets stores the filtered local datasets
        -> compliteness stores their compliteness_ratio
    '''
    filtered_local_datasets,compliteness = filter_dataset(local_datasets,local_key_code,local_metadata,compliteness_ratio)

    result = {}
    if compliteness == {}:
        return "Need to infer"
    else:
        for dataset_name, dataset in filtered_local_datasets.items():
            result[dataset_name] = dataset[var_col_names[dataset_name]]
        return result, compliteness
            
fake_data = pd.DataFrame({"absurd_column_name":[1,2,3,4,5],"another_absurd_column_name":[1,1,1,1,1],"yet_another_col_name":["FN","FN","FN","FN","FN"],"data":[1,2,3,4,5]})

datasets = {"fake_data":fake_data}

keys = {
    "polling_station":{
        "municipality_code_insee":[11,12,7],
        "BDV_code":[1,1,3]
        },
    "candidate":{
        "candidate_id":[1,4,3],
        "nuance_initials":["FN","FF","FF"]
        }
    }   

tag =  {
    "test":"test"       
    }

print(request(tag, keys, 0.1))