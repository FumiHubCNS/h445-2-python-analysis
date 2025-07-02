"""!
@file read_xcfg.py
@version 1
@author Fumitaka ENDO
@date 2025-07-01T22:10:22+09:00
@brief template text
"""
import argparse
import pathlib
import numpy as np
import h445_2_python_ana.catm.h445_utilities as h445util
import catmlib.util as util
import matplotlib.pyplot as plt

this_file_path = pathlib.Path(__file__).parent

def plot_threshold_with_pads( padsinfo=None, listName='global_threshold', 
                             showflag=True, savepath=None, 
                             legendFlag=False, title_name='global_threshold'):

    if padsinfo is None:
        print("Input of pad information is invalid. skip plot process")
        return None
    
    data_list = getattr(padsinfo[1], listName, None) 

    if data_list is None:
        print("Input of list name is invalid. skip plot process")
        return None

    threshold_dict = h445util.classify_indices(data_list)
    key_list = list(threshold_dict.keys())
    sorted_indices = np.argsort(key_list)

    value_list = list(threshold_dict.values())

    ids = []
    lebels = []
    colors = []

    total_combinations = len(value_list)

    # print(key_list,value_list)

    colormap = plt.cm.tab20c

    for i in range(total_combinations):
        ids.append(value_list[sorted_indices[i]])
        lebels.append(key_list[sorted_indices[i]])
    
        normalized_index = ( i ) / (total_combinations)  
        rgba = colormap(normalized_index)  # RGBA値
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        )
        colors.append(hex_color)
    
    x_range = (260, -260)
    y_range = (-100, 100)
    z_range = (-270, 220)

    id_list     = [[],ids,[]]
    label_list  = [[],lebels,[]]
    user_colors = [colors,colors,colors]
    util.catmviewer.plot_2d_categories( z_range, x_range, y_range, *padsinfo, *id_list, *label_list, showflag, savepath, user_colors, legendFlag, title_name)

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("-i","--input-path", help="input path of get config file", type=str, default="configure-external_soap1.xcfg")
	parser.add_argument("-o","--output-path", help="output path", type=str, default="data/kr80/padcalib/get_config_map.txt")
	parser.add_argument("-po","--pickle-output", help="padarray saving path", type=str, default="data/kr80/padcalib/catm-map-get-readoutpad.pkl")

	parser.add_argument("-rf", "--read-flag", help="flag of read config file", action="store_true")
	parser.add_argument("-wf", "--write-flag", help="flag of wire read file", action="store_true")
	parser.add_argument("-pf", "--plot-flag", help="flag of plot drawing", action="store_true")
	parser.add_argument("-sf", "--save-flag", help="flag of plot saving", action="store_true")

	args = parser.parse_args()

	maps_path, base_path, homepage_path = h445util.load_maps_path()
	get_config_path = h445util.load_get_config_path()

	input_path: str  = get_config_path + "/" + args.input_path
	output_path: str = base_path + "/" + args.output_path
	pkl_path: str = base_path + "/" + args.pickle_output

	read_flag:  bool = args.read_flag
	write_flag: bool = args.write_flag
	plot_flag:  bool = args.plot_flag
	save_flag:  bool = args.save_flag


	save_path = homepage_path if save_flag else None

	print(input_path)
	print(output_path)

	# check write method
	if write_flag:
		h445util.write_text(input_path, output_path)

	# check read method
	if read_flag:
		data = h445util.read_text(output_path)
		print(data)
    
	if plot_flag:
		data = h445util.read_text(output_path)
		pd_global = data[ (data['cobo']=='*') & (data['asad']=='*') & (data['aget']=='*') & (data['channel']=='*')]  
		pd_g_chan = data[ (data['cobo']=='*') & (data['asad']=='*') & (data['aget']=='*') & (data['channel']!='*')]
		pd_e_chan = data[ (data['cobo']!='*') & (data['asad']!='*') & (data['aget']!='*') & (data['channel']!='*') ]

		pd_g_chan = pd_g_chan.reset_index(drop=True)
		pd_e_chan = pd_e_chan.reset_index(drop=True)

		padsinfo  = h445util.get_padarrays_with_map(*maps_path)
		
		# global_threshold	LSB_threshold
		padsinfo[1].global_threshold = [pd_global['global_threshold'][0]] * len(padsinfo[1].ids)
		padsinfo[1].LSB_threshold    = [pd_global['LSB_threshold'][0]]    * len(padsinfo[1].ids)


		for i in range(len(pd_g_chan)):
			matching_indices  = h445util.get_matching_indices(padsinfo[1],-1,-1,-1,int(pd_g_chan['channel'][i]))

			for j in range(len(matching_indices)):
				if pd_g_chan['global_threshold'][i] >= 0:
					padsinfo[1].global_threshold[matching_indices[j]] = pd_g_chan['global_threshold'][i]
				if pd_g_chan['LSB_threshold'][i] >= 0:
					padsinfo[1].LSB_threshold[matching_indices[j]] = pd_g_chan['LSB_threshold'][i]

		for i in range(len(pd_e_chan)):
			# print( (pd_e_chan['cobo'][i]), (pd_e_chan['asad'][i]), (pd_e_chan['aget'][i]), (pd_e_chan['channel'][i]))
			matching_indices  = h445util.get_matching_indices(
									padsinfo[1],
									int(pd_e_chan['cobo'][i]),
									int(pd_e_chan['asad'][i]),
									int(pd_e_chan['aget'][i]),
									int(pd_e_chan['channel'][i])
									)
			
			if len(matching_indices) > 0:
				if pd_e_chan['global_threshold'][i] >= 0: 
					padsinfo[1].global_threshold[matching_indices[0]] = pd_e_chan['global_threshold'][i] 

				if pd_e_chan['LSB_threshold'][i] >= 0: 
					padsinfo[1].LSB_threshold[matching_indices[0]] = pd_e_chan['LSB_threshold'][i]


		result = h445util.classify_indices(padsinfo[1].LSB_threshold)
		print(result.keys())

		plot_flag = False if save_flag else True

		plot_threshold_with_pads(padsinfo=padsinfo, listName='LSB_threshold', legendFlag=True, title_name='LSB_threshold', showflag=plot_flag, savepath=save_path)

		if pkl_path:
			h445util.save_class_object(padsinfo[1], pkl_path)

if __name__ == "__main__":
	main()
	
	