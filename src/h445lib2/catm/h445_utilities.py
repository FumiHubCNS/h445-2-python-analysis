"""!
@file h445_utilities.py
@version 2
@author Fumitaka ENDO
@date 2025-07-02T23:51:31+09:00
@brief template text
"""
import pickle
import pathlib
import operator
import pandas as pd
import catmlib.util as catutil
import catmlib.readoutpad.catm as catm
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import reduce

this_file_path = pathlib.Path(__file__).parent

def load_get_config_path():
	toml_input = this_file_path / '../../../parameters.toml'
	toml_config = catutil.dataforming.read_toml_file(toml_input)
		
	return catutil.dataforming.expand_environment_variables( toml_config["analysis"]["environment"]["get-config"] )
		
def load_parameters_toml(tomlfilepath='../../../parameters.toml'):
	toml_input = this_file_path / tomlfilepath
	toml_config = catutil.dataforming.read_toml_file(toml_input)
		
	analysis_directory_path = toml_config["analysis"]["environment"]["analysis-directory"] 
	base_path = catutil.dataforming.expand_environment_variables(analysis_directory_path)	
	homepage_path = catutil.dataforming.expand_environment_variables(
		toml_config["analysis"]["environment"]["output"]["homepage"] 
		+ "/figure/figure001.png"
	)
	
	return toml_config, base_path, homepage_path
		
def load_maps_path():
	toml_input = this_file_path / '../../../parameters.toml'
	toml_config = catutil.dataforming.read_toml_file(toml_input)
		
	beam_tpc_file_path = toml_config["analysis"]["mapfile"]["beam-tpc"]["file"]
	recoil_tpc_file_path = toml_config["analysis"]["mapfile"]["recoil-tpc"]["file"]
	ssd_file_path = toml_config["analysis"]["mapfile"]["ssd"]["file"]
	analysis_directory_path = toml_config["analysis"]["environment"]["analysis-directory"] 

	base_path = catutil.dataforming.expand_environment_variables(analysis_directory_path)	
	homepage_path = catutil.dataforming.expand_environment_variables(
		toml_config["analysis"]["environment"]["output"]["homepage"] 
		+ "/figure/figure001.png"
	)	

	maps_path = [
		catutil.dataforming.expand_environment_variables( analysis_directory_path + "/" + beam_tpc_file_path ),
		catutil.dataforming.expand_environment_variables( analysis_directory_path + "/" + recoil_tpc_file_path ),
		catutil.dataforming.expand_environment_variables( analysis_directory_path + "/" + ssd_file_path )
	]
		
	return maps_path, base_path, homepage_path

def get_padarrays_with_map(beam_tpc_map_path=None,recoil_tpc_map_path=None, ssd_map_path=None):

	if (beam_tpc_map_path is None) or (recoil_tpc_map_path is None) or (ssd_map_path is None):
		print(f"map file path is none")
		return

	map_lists=[
		pd.read_csv(beam_tpc_map_path, sep=r'\s+', header=None),
		pd.read_csv(recoil_tpc_map_path, sep=r'\s+', header=None),
		pd.read_csv(ssd_map_path, sep=r'\s+', header=None)
	]

	pad_list = [
		catm.get_beam_tpc_array(),
		catm.get_recoil_tpc_array(),
		catm.get_ssd_array()
	]

	# init list to set GET electronics channel map 
	for i in range(len(pad_list)):
		pad_list[i].cobos    = [None] * len(pad_list[i].ids)
		pad_list[i].asads    = [None] * len(pad_list[i].ids)
		pad_list[i].agets    = [None] * len(pad_list[i].ids)
		pad_list[i].channels = [0] * len(pad_list[i].ids)

	for i in range(len(map_lists)):
		for j in range(len(map_lists[i])):

			id_ij   = map_lists[i][map_lists[i].columns[1]][j]
			cobo_ij = map_lists[i][map_lists[i].columns[3]][j]
			asad_ij = map_lists[i][map_lists[i].columns[4]][j]
			aget_ij = map_lists[i][map_lists[i].columns[5]][j]
			chan_ij = map_lists[i][map_lists[i].columns[6]][j]

			for k in range(len(pad_list)):
				index_value = get_index_from_value( pad_list[i].ids, id_ij)

				if index_value is not None:
					pad_list[i].cobos[index_value]    = int(cobo_ij)
					pad_list[i].asads[index_value]    = int(asad_ij)
					pad_list[i].agets[index_value]    = int(aget_ij)
					pad_list[i].channels[index_value] = int(chan_ij)
					break

	return pad_list

def get_index_from_value(data, value):
	if value in data:
		return data.index(value)
	else:
		return None

def classify_indices(values):
	index_dict = defaultdict(list)
	for idx, val in enumerate(values):
		index_dict[val].append(idx)
	return dict(index_dict)

def get_list_attribute(obj, base_name: str):
	attr_name = base_name.lower() + "s"
	if not hasattr(obj, attr_name):
		raise AttributeError(f"Object has no attribute '{attr_name}'")

	value = getattr(obj, attr_name)
	if not isinstance(value, list):
		raise TypeError(f"Attribute '{attr_name}' exists but is not a list")

	return value

def get_xcfg_tree(input_path=None):
	if input_path is None:
		print("input path is none")
		return
	tree = ET.parse(input_path)
	root = tree.getroot()
	return root

def get_xcfg_node(input_path=None, node_id_name='CoBo'):
	if input_path is None:
		print("input path is none")
		return
	root = get_xcfg_tree(input_path)
	cobo_node = root.find(f".//Node[@id='{node_id_name}']")
	return cobo_node

def get_xcfg_instance(node=None, instance_id_name='*'):
	if node is None:
		print("input path is none")
		return
	instance = node.find(f".//Instance[@id='{instance_id_name}']")
	return instance

def get_xcfg_block(instance=None, label_name='AsAd', block_name='0' ):
	if instance is None:
		print("input path is none")
		return
	block_data = instance.find(f".//{label_name}[@id='{block_name}']")
	return block_data

def print_xcfg_tree(element=None, indent=0, indent_label=1):
	if element is None:
		print("input path is none")
		return
	if indent == indent_label:
		print("  " * indent + f"{element.tag}: {element.attrib}")

	for child in element:
		print_xcfg_tree(child, indent + 1, indent_label)

def write_text(input_path=None, output_path=None):

	if input_path is None:
		print("input path is none")
		return

	if output_path is None:
		print("input path is none")
		return

	cobo_node = get_xcfg_node(input_path)

	cobos_list = ['*','0','1','2','3']
	asads_list = ['*','0','1','2','3']
	agets_list = ['*','0','1','2','3']

	threshold_map = []

	for i in range(len(cobos_list)):
		instance = get_xcfg_instance(cobo_node, cobos_list[i])
		if instance is not None:

			cobo_id_block = instance.find(f"./Module/coboId")
			cobo_id = cobo_id_block.text if cobo_id_block is not None else -1

			for j in range(len(asads_list)):
				asad = get_xcfg_block(instance,'AsAd', asads_list[j])
				if asad is not None:

					for k in range(len(agets_list)):
						aget = get_xcfg_block(asad,'Aget', agets_list[k])
						if aget is not None:

							global_threshold_block = aget.find(f"./Global/Reg1/GlobalThresholdValue")
							global_threshold = global_threshold_block.text if (global_threshold_block is not None) else -1

							channels = aget.findall('channel')

							for l in range(len(channels)):
								lsb_threshold_block = channels[l].find(f"./LSBThresholdValue")
								lsb_threshold = lsb_threshold_block.text if lsb_threshold_block is not None else -1

								threshold_map.append( 
									[
										cobos_list[i],
										asads_list[j], 
										agets_list[k],
										channels[l].attrib.get("id"),
										cobo_id,
										global_threshold,
										lsb_threshold
									] 
								)
							
	pd_label_name = ['cobo', 'asad', 'aget', 'channel','coboId','global_threshold','LSB_threshold']
	data = pd.DataFrame(threshold_map, columns=pd_label_name)     
	data.to_csv(output_path, index=False, sep='\t')
	print(f"save csv to {output_path}")

def read_text(input_path=None):

	if input_path is not None:
		data = pd.read_csv(input_path,sep='\t')
		return data
	else:
		print("input path is none")
    
def get_matching_indices(data, cobo_value, asad_value, aget_value, channel_value):
	matching_indices = []

	for i in range(len(data.ids)):
		if cobo_value >= 0:
			cobo_flag = data.cobos[i] == cobo_value
		else:
			cobo_flag = True

		if asad_value >= 0:
			asad_flag = data.asads[i] == asad_value
		else:
			asad_flag = True

		if aget_value >= 0:
			aget_flag = data.agets[i] == aget_value
		else:
			aget_flag = True

		if channel_value >= 0:
			chan_flag = data.channels[i] == channel_value
		else:
			chan_flag = True
		
		if chan_flag * aget_flag * asad_flag * cobo_flag == True:
			matching_indices.append(i)        

	return matching_indices

def match_filter(values, ref_val):
    return [(v == ref_val) if ref_val >= 0 else True for v in values]

def get_map_sort_result_list(pad_list=None, ref_cobo=-1, ref_asad=-1, ref_aget=-1, ref_chan=-1):
    if pad_list is not None:
        result_list = []

        for i in range(len(pad_list)):
            cobos = pad_list[i].cobos
            asads = pad_list[i].asads
            agets = pad_list[i].agets
            chans = pad_list[i].channels

            sorted_cobos = match_filter(cobos, ref_cobo)
            sorted_asads = match_filter(asads, ref_asad)
            sorted_agets = match_filter(agets, ref_aget)
            sorted_chans = match_filter(chans, ref_chan)

            result = [reduce(operator.mul, values) for values in zip(sorted_cobos, sorted_asads, sorted_agets, sorted_chans)]
            result_list.append(result)
    
    else:
        result_list = None
    
    return result_list

def save_class_object(data=None, output=None):
	if (data is None) or (output is None):
		print(f"input object or output path is none")
	
	else:
		with open(output, "wb") as f:
			pickle.dump(data, f)

def load_class_object(input=None):
	if input is None:
		print(f"input object or output path is none")
	
	else:
		with open(input, "rb") as f:
			data = pickle.load(f)
		
		return data

def main():
	# check map mergeing process  
	maps_path, base_path, homepage_path = load_maps_path()
	print(f"analysis directory path : {base_path}")
	for i in range(len(maps_path)):
		print(f"path : {maps_path[i]}")
	
	pad_list = get_padarrays_with_map(*maps_path)
	print(f"number of pad array class : {len(pad_list)}")

	for i in range(3):
		print(f"check get config at pad array {i} : ID={pad_list[i].ids[0]}, ({pad_list[i].cobos[0]}, {pad_list[i].asads[0]}, {pad_list[i].agets[0]}, {pad_list[i].channels[0]}), pads={len(pad_list[i].ids)}")

	#xcfg test
	get_config_path = load_get_config_path()
	print(get_config_path)

if __name__ == "__main__":
	main()
