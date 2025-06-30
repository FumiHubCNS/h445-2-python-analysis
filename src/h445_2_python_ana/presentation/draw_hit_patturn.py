"""!
@file draw_hit_patturn.py
@version 1
@author Fumitaka ENDO
@date 2025-06-30T21:00:32+09:00
@brief draw hit patturn for presentaion
"""
import argparse
import pandas as pd
import numpy as np
import pathlib
import catmlib.util as catutil
import catmlib.readoutpad.catm as catm

this_file_path = pathlib.Path(__file__).parent

def check_and_generate_directory(path_str):

    dir_path = pathlib.Path(path_str)
    
    if not dir_path.exists():
        print(f"directory does not exist. generate directory: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=True)

def get_padinfo():
	padinfo = []
	padinfo.append( catm.get_beam_tpc_array() )
	padinfo.append( catm.get_recoil_tpc_array() )
	padinfo.append( catm.get_ssd_array() )
	
	return padinfo

def load_data(input_file_path_name, event_list_path_name):

	toml_input = this_file_path / '../../../parameters.toml'
	toml_config = catutil.dataforming.read_toml_file(toml_input)

	base_path_info = toml_config["analysis"]["environment"]
	base_input_path = base_path_info["input"]["base"]
	base_output_figure_path = toml_config["analysis"]["environment"]["output"]["figure"]

	analysis_directory_path = base_path_info["analysis_directory"] 	

	input_path  = analysis_directory_path + "/" + base_input_path + "/" + input_file_path_name
	event_path  = analysis_directory_path + "/" + base_input_path + "/" + event_list_path_name
	output_path = analysis_directory_path + "/" + base_output_figure_path + "/"

	input_path = catutil.dataforming.expand_environment_variables(input_path)
	event_path = catutil.dataforming.expand_environment_variables(event_path) 
	output_path = catutil.dataforming.expand_environment_variables(output_path) 
	homepage_path = catutil.dataforming.expand_environment_variables(toml_config["analysis"]["environment"]["output"]["homepage"])

	path_flag = pathlib.Path(input_path)
	if not path_flag.is_file():
		print(f"input data file does not exist")
		return
	
	path_flag = pathlib.Path(event_path)
	if not path_flag.is_file():
		print(f"event list file does not exist")
		return

	raw_data = pd.read_csv(input_path)
	target_numbers = catutil.dataforming.load_numbers(event_path)
	output_path_list =  [output_path, homepage_path]

	return raw_data, target_numbers, output_path_list


def plot_track(
		evtnum=0, padinfo=None, raw_data=None,
		drawflag2d=False ,savepath2d=None,
		drawflag3d=False ,savepath3d=None
	):

	if padinfo is None:
		print("pad array list is none")
		return 

	if raw_data is not None:

		anaflag = raw_data.tpcsiflag[evtnum]
			
		# recoil tpc hit
		rox = catutil.dataforming.str_to_array(raw_data.recoil_hit_pattern_x[evtnum])
		roy = catutil.dataforming.str_to_array(raw_data.recoil_hit_pattern_y[evtnum])
		roz = catutil.dataforming.str_to_array(raw_data.recoil_hit_pattern_z[evtnum])
		rid = catutil.dataforming.str_to_array(raw_data.recoil_hit_pattern_id[evtnum])

		# beam tpc hit
		box = catutil.dataforming.str_to_array(raw_data.beam_hit_pattern_x[evtnum])
		boy = catutil.dataforming.str_to_array(raw_data.beam_hit_pattern_y[evtnum])
		boz = catutil.dataforming.str_to_array(raw_data.beam_hit_pattern_z[evtnum])
		bid = catutil.dataforming.str_to_array(raw_data.beam_hit_pattern_id[evtnum])

		# beam tpc hit
		sox = catutil.dataforming.str_to_array(raw_data.ssd_hit_pattern_x[evtnum])
		soy = catutil.dataforming.str_to_array(raw_data.ssd_hit_pattern_y[evtnum])
		soz = catutil.dataforming.str_to_array(raw_data.ssd_hit_pattern_z[evtnum])
		sid = catutil.dataforming.str_to_array(raw_data.ssd_hit_pattern_id[evtnum])

		# beam track outside magnet
		bmx = catutil.dataforming.str_to_array(raw_data.endpoint_of_beam_track_outside_magnet[evtnum])
		bmv = catutil.dataforming.str_to_array(raw_data.diretion_of_beam_track_outside_magnet[evtnum])
		btm = catutil.dataforming.str_to_array(raw_data.beam_track_in_magnet[evtnum])

		bpos = catutil.catmviewer.calculate_extrapolated_position(bmx, bmv, -255)
		bx = np.array([bpos[0],bmx[0]])
		by = np.array([bpos[1],bmx[1]])
		bz = np.array([bpos[2],bmx[2]])

		bpos2d = catutil.catmviewer.calculate_extrapolated_position(bmx, bmv, -270)
		bx2d = np.array([bpos2d[0],bmx[0]])
		by2d = np.array([bpos2d[1],bmx[1]])
		bz2d = np.array([bpos2d[2],bmx[2]])

		shifty = bpos[1]-np.mean(boy)

		# beam track in magnet
		
		omega = btm[2]
		bmx = bmx*1e-3
		bmv = catutil.catmviewer.calculate_unit_vector(bmv)
		bmv = bmv*btm[1]
		t = np.linspace(0, 2.5e-9, 101)
		bmx, bmy, bmz = catutil.catmviewer.calculate_track_dipole_magnet_analytical_solution(bmv,bmx,omega,t)

		# recoil track in magnet
		rmx = catutil.dataforming.str_to_array(raw_data.recoil_track_x_in_magnet[evtnum])
		rmy = catutil.dataforming.str_to_array(raw_data.recoil_track_y_in_magnet[evtnum])
		rmz = catutil.dataforming.str_to_array(raw_data.recoil_track_z_in_magnet[evtnum])
		
		# recoil track outside magnet
		sotr = catutil.dataforming.str_to_array(raw_data.startpoint_of_recoil_track_outside_magnet[evtnum])
		eotr = catutil.dataforming.str_to_array(raw_data.endpoint_of_recoil_track_outside_magnet[evtnum])
		votr = eotr - sotr

		rmax_index = np.argmax(rox)
		stes = catutil.catmviewer.calculate_extrapolated_position(sotr, votr, rox[rmax_index], 0)
		etss = catutil.catmviewer.calculate_extrapolated_position(sotr, votr, 255*rox[rmax_index]/abs(rox[rmax_index]), 0)
		etss2d = catutil.catmviewer.calculate_extrapolated_position(sotr, votr, 280*rox[rmax_index]/abs(rox[rmax_index]), 0)
		sx = np.array([rmx[0],etss[0]])
		sy = np.array([rmy[0],etss[1]])
		sz = np.array([rmz[0],etss[2]])
		sx2d = np.array([rmx[0],etss2d[0]])
		sy2d = np.array([rmy[0],etss2d[1]])
		sz2d = np.array([rmz[0],etss2d[2]])

		verr = catutil.dataforming.str_to_array(raw_data.vertex_point_recoil[evtnum])

		# vertex draw option
		rindex = catutil.catmviewer.find_nearest_index(rmy, verr[1])
		bindex = catutil.catmviewer.find_nearest_index(bmz, 150e-3)

		px = [rox, box        ]
		py = [roy, boy+shifty ]
		pz = [roz, boz-255    ]
		
		x_range = (260, -260)
		y_range = (-100, 100)
		z_range = (-270, 220)
			
		if (drawflag3d == True) or (savepath3d is not None):
			lx = [rmx[:rindex], bmx[:bindex]*1e3,bx,sx]
			ly = [rmy[:rindex], bmy[:bindex]*1e3,by,sy]
			lz = [rmz[:rindex], bmz[:bindex]*1e3,bz,sz]
			catutil.catmviewer.plot_3d_trajectory(pz, px, py, lz, lx, ly, z_range, x_range, y_range, *padinfo, bid, rid, sid, anaflag, drawflag3d, savepath3d)

		if (drawflag2d == True) or (savepath2d is not None):
			lx = [rmx[:rindex], bmx[:bindex]*1e3,bx,sx2d]
			ly = [rmy[:rindex], bmy[:bindex]*1e3,by,sy2d]
			lz = [rmz[:rindex], bmz[:bindex]*1e3,bz,sz2d]
			catutil.catmviewer.plot_2d_trajectory(pz, px, py, lz, lx, ly, z_range, x_range, y_range, *padinfo, bid, rid, sid, anaflag, drawflag2d, savepath2d)
	
	else:
		print("input tracking data is invalid")

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("-ml","--max-loop", help="select maximum loop", type=int, default=1)
	parser.add_argument("-i","--input-path", help="input path of data", type=str, default="track-csv-output/output-woSi.txt")
	parser.add_argument("-e","--event-list", help="input path of event list", type=str, default="track-csv-output/track-list-woSi.txt")
	parser.add_argument("-o2d","--output-2d-directory", help="output direcory name of 2d plot", type=str, default="track2DwoSiv0")
	parser.add_argument("-o3d","--output-3d-directory", help="output direcory name of 3d plot", type=str, default="track3DwoSi")
	
	parser.add_argument("-fd2d","--flag-2d-draw", help="show 2d track plot", action="store_true")
	parser.add_argument("-fd3d","--flag-3d-draw", help="show 3d track plot", action="store_true")
	parser.add_argument("-fs2d","--flag-2d-save", help="save 2d track plot", action="store_true")
	parser.add_argument("-fs3d","--flag-3d-save", help="save 3d track plot", action="store_true")
	parser.add_argument("-fh","--flag-homepage", help="flag of saveing plot in homepage", action="store_true")
	
	args = parser.parse_args()

	max_loop: int = args.max_loop
	drawflag2d: bool = args.flag_2d_draw
	drawflag3d: bool = args.flag_3d_draw
	saveflag2d: bool = args.flag_2d_save
	saveflag3d: bool = args.flag_3d_save
	savehomepage: bool = args.flag_homepage
	input_file_path_name: str = args.input_path 
	event_list_path_name: str = args.event_list
	output_directory_2d: str = args.output_2d_directory
	output_directory_3d: str = args.output_3d_directory

	padinfo = get_padinfo()
	raw_data, target_numbers, output_path_list = load_data(input_file_path_name, event_list_path_name)

	if not pathlib.Path(output_path_list[0]).exists():
		print(f"[warning] base path of output directory does not exist. this method will be error when you save figures in analysis directory")
	
	if not pathlib.Path(output_path_list[1]).exists():
		print(f"[warning] base path of output directory does not exist. this method will be error when you save figures in homepage")
    
	counts = 0
	
	for evtnum in range(max_loop):
		if evtnum in target_numbers or len(target_numbers) == 0:

			savepath2d = None
			savepath3d = None
			
			if saveflag2d:
				savepath2d = output_path_list[0] + output_directory_2d + "/track{:03}".format(evtnum) + '.png'

				if savehomepage:
					savepath2d = output_path_list[1] + "/figure001.png"
				else:
					if counts == 0:
						check_and_generate_directory(output_path_list[0] + output_directory_2d)

			if saveflag3d:
				savepath3d = output_path_list[0] + output_directory_3d + "/track{:03}".format(evtnum) + '.png'
				
				if savehomepage:
					savepath3d = output_path_list[1] + "/figure002.png"
				else:
					if counts == 0:
						check_and_generate_directory(output_path_list[0] + output_directory_2d)

			plot_track(evtnum, padinfo, raw_data, drawflag2d, savepath2d, drawflag3d, savepath3d)
			counts += 1
		else:
			print(f"Skipping logic for {evtnum}")


if __name__ == "__main__":
	main()