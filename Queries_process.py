import urllib.request, json 
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn import linear_model

# Import dataset
def get_dataset(dataset_name:str,metadata:dict):
    source = metadata[dataset_name]["get"]["source"]
    if source == "local":
        extension = metadata[dataset_name]["get"]["extension"]
        if extension == '.csv':
            return pd.read_csv("/Users/julien/QUORUM/Priorisation2/DataProvider/Datasets/"+dataset_name+extension, sep=";")
        else:
           return pd.DataFrame() 
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
def get_keys_correlation(key:str):
    source = "/Users/julien/QUORUM/Priorisation2/DataProvider/Keys/"+key+".json"
    with open(source) as f:
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
                dataset = get_dataset(dataset_name,metadata)
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
            datasets[dataset_name] = get_dataset(dataset_name,metadata)
    return datasets, new_metadata



# Convert dataset to a specific set of keys
def convert_dataset_to_key_code_version(key_codes:dict, datasets:dict, metadata:dict):
    result = {}
    keys_correlation = {}
    for key in key_codes:
        keys_correlation[key] = get_keys_correlation(key)
    key_codes = deepcopy(key_codes)
    datasets = deepcopy(datasets)
    metadata = deepcopy(metadata)
    new_metadata = {}

    for dataset_name, dataset in datasets.items():
        # Change key version if needed
        if set(metadata[dataset_name]["keys"].keys()) == set(key_codes):
            new_metadata[dataset_name] = deepcopy(metadata[dataset_name])

            for key in metadata[dataset_name]["keys"]:
                if set(key_codes[key].keys()) != set(metadata[dataset_name]["keys"][key].keys()):

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

                else:
                    for key_element, col_name in metadata[dataset_name]["keys"][key].items():
                        dataset.rename(columns={col_name:key_element},inplace=True)
                    new_metadata[dataset_name]["keys"][key] = {}
                    new_version = metadata[dataset_name]["keys"][key].keys()
                    for key_element in new_version:
                        new_metadata[dataset_name]["keys"][key][key_element] = key_element

            result[dataset_name] = dataset           
                

    return (result,new_metadata)

# Returns how muck keys there are in the datasets
def filter_dataset(datasets:dict, keys:dict, metadata:dict, compliteness_ratio=0):

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

# Add a column to a dataset following the keys on the right
def add_column(new_column:object,new_keys:object,existing_df:object):

    new_column = deepcopy(new_column)
    new_keys = deepcopy(new_keys)
    existing_df = deepcopy(existing_df)
    existing_keys = existing_df[new_keys.columns]
    new_keys = new_keys.reindex(existing_keys.columns, axis=1)
    result = pd.Series(np.nan,index=existing_df.index)

    for i in range(len(existing_keys)):
        j = 0
        while not all(existing_keys.loc[[i]].reset_index(drop=True)==new_keys.loc[[j]].reset_index(drop=True)):
            j += 1
            if j == len(new_keys):
                break
        if j != len(new_keys):
            result.loc[[i]] = deepcopy(new_column.loc[[j]])
        
    return result

# Look for features (with the same tags in local and training datasets). Returns local_features and training_features
def lookup_features(local_metadata:dict, local_datatsets:dict, training_metadata:dict, training_datasets:dict, compliteness:dict):

    local_metadata = deepcopy(local_metadata)
    training_metadata = deepcopy(training_metadata)
    local_datasets = deepcopy(local_datatsets)
    training_datasets =  deepcopy(training_datasets)
    metadata = {}
    local_features = pd.DataFrame()
    training_features = pd.DataFrame()
    
    number_features = 0

    #Looking for features with the same tag
    for training_dataset_name in training_metadata:
        for training_feature, training_feature_metadata in training_metadata[training_dataset_name]["raw_data"].items():
            for local_dataset_name in local_metadata:
                for local_feature, local_feature_metadata in local_metadata[local_dataset_name]["raw_data"].items():
                    if are_tags_equal(local_feature_metadata["tags"], training_feature_metadata["tags"]):

                        # Getting the columns from the datasets
                        local_key_columns = pd.DataFrame()
                        for key in local_metadata[local_dataset_name]["keys"].values():
                            for key_element, column_name in key.items():
                                local_key_columns[key_element] = local_datasets[local_dataset_name][column_name]
                        local_feature_column = local_datasets[local_dataset_name][local_feature]
                        local_feature_column.columns = ["feature_"+str(number_features)]


                        training_key_columns = pd.DataFrame()
                        for key in training_metadata[training_dataset_name]["keys"].values():
                            for key_element, column_name in key.items():
                                training_key_columns[key_element] = training_datasets[training_dataset_name][column_name]
                        training_feature_column = training_datasets[training_dataset_name][training_feature]
                        training_feature_column.columns = ["feature_"+str(number_features)]

                        # Pasting keys
                        if number_features == 0:
                            local_features = deepcopy(local_key_columns)
                            training_features = deepcopy(training_key_columns)
                        
                        # Adding feature columns, updtaing metadata and compliteness
                        local_features["feature_"+str(number_features)] = add_column(local_feature_column,local_key_columns,local_features)
                        training_features["feature_"+str(number_features)] = add_column(training_feature_column,training_key_columns,training_features)
                        metadata["feature_"+str(number_features)]={"local": {local_dataset_name:local_metadata[local_dataset_name]}}
                        metadata["feature_"+str(number_features)]["training"] = {training_dataset_name:training_metadata[training_dataset_name]}
                        metadata["feature_"+str(number_features)]["metadata"] = compliteness[local_dataset_name]
                        number_features += 1


    return local_features, training_features, metadata

def convert_dataset_to_keys():
    return pd.DataFrame()

#Get metadata from datasets with the right dataset tags
def get_datasets_with_dataset_tag(metadata, dataset_tags):

    new_metadata = deepcopy(metadata)
    for dataset_name in new_metadata.keys():
        for tag, tag_value in dataset_tags.items():
            if tag not in metadata[dataset_name].keys():
                del new_metadata[dataset_name]
            elif metadata[dataset_name][tag] != tag_value:
                del new_metadata[dataset_name]
    
    return new_metadata



# Basic request to get all datasets with the right tag with the specified keys
def request(var_tag:dict, local_key_code:dict, compliteness_ratio=0, dataset_tags={}):

    metadata = get_metadata()

    metadata = get_datasets_with_dataset_tag(metadata, dataset_tags)

    '''
    Look for datasets with a variable with the right tag
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
        print("Need to infer")
        return var_datasets,var_metadata,var_col_names

    else:
        for dataset_name, dataset in filtered_local_datasets.items():

            key_col_names = []
            for key in local_metadata[dataset_name]["keys"].values():
                key_col_names += [col_name for col_name in key.values()]             

            result[dataset_name] = dataset[[var_col_names[dataset_name]]+key_col_names]

    return result, compliteness

def get_features(var_datasets:dict,var_metadata:dict,var_col_names:dict, local_key_code:dict, compliteness_ratio:float):
    
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
    Get var_key_code and var_column
    '''
    var_column = {}
    var_key_code = {}
    for var_dataset_name in var_datasets.keys():
        var_column[var_dataset_name] = pd.DataFrame()
        var_key_code[var_dataset_name] = {}
        for key in metadata[var_dataset_name]["keys"].keys():
            var_key_code[var_dataset_name][key] = {}
            for key_element, column_name in metadata[var_dataset_name]["keys"][key].items():
                var_key_code[var_dataset_name][key][key_element] = var_datasets[var_dataset_name][column_name].tolist()
                var_column[var_dataset_name][key_element] = var_datasets[var_dataset_name][column_name]
        var_column[var_dataset_name][var_col_names[var_dataset_name]] = var_datasets[var_dataset_name][var_col_names[var_dataset_name]]    

    '''
    Look for training_datasets, convert it to var_key_code
    -> training_datasets stores all possible datasets with the same keys as var_datasets => Possible training_features
    -> training_metadata stores their metadata
    '''

    training_datasets = {}
    training_metadata = {}
    training_compliteness = {}
    for var_dataset_name in var_datasets.keys():
        training_datasets[var_dataset_name], training_metadata[var_dataset_name] = get_dataset_from_key_code(var_key_code[var_dataset_name],metadata)
        training_datasets[var_dataset_name], training_metadata[var_dataset_name] = convert_dataset_to_key_code_version(var_key_code[var_dataset_name],training_datasets[var_dataset_name],training_metadata[var_dataset_name])
        training_datasets[var_dataset_name], training_compliteness[var_dataset_name] = filter_dataset(training_datasets[var_dataset_name],var_key_code[var_dataset_name],training_metadata[var_dataset_name])
        
    '''
    Look for features
    '''

    metadata_var = {}
    local_features_datasets = {}
    training_features_and_label_datasets = {}
    for var_dataset_name in var_datasets.keys():
        local_features_datasets[var_dataset_name],  training_features_and_label_datasets[var_dataset_name], metadata_var[var_dataset_name]= \
        lookup_features(local_metadata,filtered_local_datasets,training_metadata[var_dataset_name],training_datasets[var_dataset_name],compliteness)
        training_features_and_label_datasets[var_dataset_name]["label"] = add_column(var_column[var_dataset_name][var_col_names[var_dataset_name]],var_column[var_dataset_name].drop(var_col_names[var_dataset_name], axis=1), training_features_and_label_datasets[var_dataset_name])        

    return local_features_datasets, training_features_and_label_datasets, metadata_var

# Apply functions before inference
def apply_functions(dataset:object, functions:dict):

    # Check if columns match
    if list(functions.keys()) not in dataset.columns:
        raise Exception("Columns do not match")

    for column_name in functions.keys():
        dataset[column_name] = getattr(*functions[column_name])(dataset[column_name])

    return dataset
        


def infer(local_features:object, training_features_and_label:object, model="linear_regression", parameters = {}):

    # Drop keys from local_features
    feature_col_names = [col_name for col_name in local_features.columns if col_name.startswith("feature_")]
    local_label_and_keys = pd.DataFrame()

    # Need to drop/fill NaNs

    if model == "linear_regression":
        # Train model on training_features
        regr = linear_model.LinearRegression()
        training_features = training_features_and_label[feature_col_names]
        training_label = training_features_and_label["label"]
        regr.fit(training_features, training_label)

        # Use on local_data
        local_label = regr.predict(local_features[feature_col_names])
        local_label_and_keys = pd.concat([pd.Series(local_label, name="label"),local_features.drop(feature_col_names,axis=1)],axis = 1)

    return local_label_and_keys

      
def get_key_correlation_from_dataset(dataset_name:str, metadata:dict):

    dataset = get_dataset(dataset_name,metadata)
    dataset = dataset.rename(columns=dict(zip(dataset.columns,[x.lower() for x in dataset.columns]), inplace=True))
    for key in metadata[dataset_name]["keys"]:
        key_correlation = get_keys_correlation(key)
        version = -1
        for i in range(len(key_correlation["versions"])):
            if set(key_correlation["versions"][i].keys()) == set(metadata[dataset_name]["keys"][key].keys()):
                version = i
        if version == -1:
            key_correlation["versions"].append({})
        for key_element,colname in metadata[dataset_name]["keys"][key].items():
            if colname.lower() not in dataset.columns:
                print(colname, dataset_name)
            else:
                for x in dataset[colname.lower()].tolist():
                    if x not in key_correlation["versions"][i][key_element]:
                        key_correlation["versions"][i][key_element].append(x)
        with open("/Users/julien/QUORUM/Priorisation2/DataProvider/Keys/"+key+".json",'w') as f:
            dataa = json.dumps(key_correlation, ensure_ascii=False, indent=4, sort_keys=False)
            f.write(dataa)
            return True