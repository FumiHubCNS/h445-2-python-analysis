"""!
@file check_map_asign.py
@version 1
@author Fummitaka ENDO
@date 2025-07-01T23:54:25+09:00
@brief template text
"""
import argparse
import pathlib
import random
import h445_2_python_ana.catm.h445_utilities as h445util
import catmlib.util as util
import matplotlib.pyplot as plt

this_file_path = pathlib.Path(__file__).parent

def generate_random_html_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def check_and_return_list(raw,debugFlag=False):
    if not isinstance(raw, list):
        val = raw
        raw = []
        raw.append(val)

    if debugFlag == True:
        print(raw)

    return raw

def check_all_map(padsinfo=None, refname='cobo', showflag=False, savepath=None, legendflag=False):

	if padsinfo is None:
		print("pad array list is none")
		return 
	else:
		if len(padsinfo) != 3:
			print(f"length of pad array list should be 3. current : {len(padsinfo)}")
			return
		
	x_range = (260, -260)
	y_range = (-100, 100)
	z_range = (-270, 220)

	all_categories_data = []
	pre_user_colors=[]

	cat_labels = ["0","1","2","3","4"]
	cat_colors = ['#B844A0',"#5F75D7","#44B86B","#CCDF41","#66E5E3"]

	for i in range(len(padsinfo)):

		cat_datas = [ [] for _ in cat_labels ]
		pads_colors = []
		
		data = h445util.get_list_attribute(padsinfo[i], refname)
		data_dict = h445util.classify_indices(data)

		data_dict_keys = list(data_dict.keys())
		data_dict_vals = list(data_dict.values())

		for j in range(len(data_dict_keys)):
			padlists = []
			
			for k in range(len(data_dict_vals[j])):
				padlists.append(padsinfo[i].ids[data_dict_vals[j][k]])
			
			cat_datas[j] = padlists 

			pads_colors.append(cat_colors[int(data_dict_keys[j])])
		
		pre_user_colors.append(pads_colors)	
		all_categories_data.append(cat_datas)
	
	user_colors = [ pre_user_colors[1], pre_user_colors[0], pre_user_colors[2] ]

	util.catmviewer.plot_2d_categories( z_range, x_range, y_range, *padsinfo, *all_categories_data, cat_labels, cat_labels, cat_labels, showflag, savepath, user_colors, legendflag)

def check_position_with_cobo_adad_aget( padsinfo=None, cobo=-1, asad=-1, aget=-1, chan=-1, showflag=False, savepath=None, legendflag=False, colormapname='rainbow'):
	
	if padsinfo is None:
		print("pad array list is none")
		return 
	else:
		if len(padsinfo) != 3:
			print(f"length of pad array list should be 3. current : {len(padsinfo)}")
			return
		
	x_range = (260, -260)
	y_range = (-100, 100)
	z_range = (-270, 220)

	colormap = getattr(plt.cm, colormapname)

	cobo = check_and_return_list(cobo)
	asad = check_and_return_list(asad)
	aget = check_and_return_list(aget)
	chan = check_and_return_list(chan)

	total_combinations = len(cobo) * len(asad) * len(aget) * len(chan)

	colors = []
	ids_list = []
	labels_list = []

	ids_list = [[[] for _ in range(total_combinations)] for _ in range(len(padsinfo))]
	labels_list = [[[] for _ in range(total_combinations)] for _ in range(len(padsinfo))]

	index = 0 

	for j in range(len(cobo)):
		for k in range(len(asad)):
			for l in range(len(aget)):
				for m in range(len(chan)):
					searched_data = h445util.get_map_sort_result_list(padsinfo, cobo[j], asad[k], aget[l], chan[m])

					for i in range(len(searched_data)):
						data_dict = h445util.classify_indices(searched_data[i])
		
						ids_list[i][index] = list(data_dict.get(1, []))
						labels_list[i][index] = f"({cobo[j]}, {asad[k]}, {aget[l]}, {chan[m]})"
	
					normalized_index = index / (total_combinations - 1) 
					rgba = colormap(normalized_index)  # RGBA値
					hex_color = '#{:02x}{:02x}{:02x}'.format(
						int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
					)

					colors.append(hex_color)

					index += 1
					searched_data = []
	
	user_colors=[colors,colors,colors]
	util.catmviewer.plot_2d_categories( z_range, x_range, y_range, *padsinfo, *ids_list, *labels_list, showflag, savepath, user_colors, legendflag)

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("-t","--type", help="select method", type=int, default=0)
	parser.add_argument("-l","--label", help="list name for ploting", type=str, default="cobo")
	parser.add_argument("-cm","--colormap-name", help="select colormap", type=str, default="rainbow")

	parser.add_argument("-pf", "--plot-flag", help="flag of plot drawing", action="store_true")
	parser.add_argument("-sf", "--save-flag", help="flag of plot saving", action="store_true")
	parser.add_argument("-lf", "--legend-flag", help="flag of plot legend", action="store_true")

	args = parser.parse_args()

	check_type: int = args.type

	list_name: str  = args.label
	colormap_name: str = args.colormap_name

	plot_flag: bool = args.plot_flag
	save_flag: bool = args.save_flag
	legend_flag: bool = args.legend_flag
	
	maps_path, base_path, homepage_path = h445util.load_maps_path()
	save_path = homepage_path if save_flag else None
	padsinfo  = h445util.get_padarrays_with_map(*maps_path)
	
	if check_type == 0:
		check_all_map(padsinfo, refname=list_name, showflag=plot_flag, savepath=save_path, legendflag=legend_flag)
	
	elif check_type == 1 :
		check_position_with_cobo_adad_aget(padsinfo,cobo=[0,1,2], asad=[0,1,2], aget=-1, chan=-1, showflag=plot_flag, savepath=save_path, legendflag=legend_flag,colormapname=colormap_name)


if __name__ == "__main__":
	main()
	
	