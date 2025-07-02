"""!
@file check_pad_histograms.py
@version 1
@author Fumitaka ENDO
@date 2025-07-02T02:41:48+09:00
@brief check histogarams using map/geometrical information
"""
import os
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from iminuit import Minuit
import catmlib.util as util
import h445_2_python_ana.catm.h445_utilities as h445util

this_file_path = pathlib.Path(__file__).parent

def calculate_3d_distance(p1=[0,0,0], p2=[0,0,0], dx=0):
    for i in range(len(p1)):
        dx += np.power( p1[i] - p2[i] , 2)
    return np.sqrt(dx)

def calculate_1d_distance(p1=[0,0,0], p2=[0,0,0], i=0):
    return np.abs(p1[i] - p2[i])

def merge_padarray_and_histogram(padsinfo=None, hist_csv_path=None ):
    if padsinfo is None:
        print("pad array list is none")
        return 

    else:
        if len(padsinfo) != 3:
            print(f"length of pad array list should be 3. current : {len(padsinfo)}")
            return
        
    chargeid_csv = pd.read_csv(hist_csv_path)
    # get 1 dimension histogram from csv data
    values = chargeid_csv["NormalizedValue"].tolist()
    errors = chargeid_csv["Error"].tolist() 
    integs = chargeid_csv["Integral"].tolist()  
    ids    = chargeid_csv["XBinCenter"].tolist()

    # divide data by bin number 
    chunk_size = 0

    first_val = ids[0]
    for idx, val in enumerate(ids):
        if val != first_val:
            chunk_size = idx
            break

    split_list        = [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]
    split_list_err    = [errors[i:i + chunk_size] for i in range(0, len(errors), chunk_size)]
    split_id          = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
    split_integ       = [integs[i:i + chunk_size] for i in range(0, len(integs), chunk_size)]
    chargeid_csv_yval = chargeid_csv.YBinCenter[0:chunk_size] 

    # init and set histograme
    for i in range(len(padsinfo)):
        padsinfo[i].hist      = [None] * len(padsinfo[i].ids)
        padsinfo[i].error     = [None] * len(padsinfo[i].ids)
        padsinfo[i].bincenter = [None] * len(padsinfo[i].ids)
        padsinfo[i].hist_sum  = [None] * len(padsinfo[i].ids)

    for i in range(len(split_list)):
        for j in range(len(padsinfo)):
            
            index_value = h445util.get_index_from_value( padsinfo[j].ids, split_id[i][0])
            
            if index_value is not None:
                padsinfo[j].hist[index_value]      = split_list[i]
                padsinfo[j].error[index_value]     = split_list_err[i]
                padsinfo[j].bincenter[index_value] = chargeid_csv_yval  
                padsinfo[j].hist_sum[index_value] = split_integ[i][0]
                break

    return padsinfo

def draw_charge_1d_histogram_with_getdevice(
        padsinfo=None, cobo=-1,asad=-1, aget=-1, channel=-1,
        showflag=False, savepath=None, legendflag=False, colormapname='rainbow', homepageflag=True, savepath2=None
    ):

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

    flag_array =  h445util.get_map_sort_result_list(padsinfo, cobo, asad, aget, channel)
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111)

    colormap = getattr(plt.cm, colormapname)
    color_index = 0
    total_draw_counts = 0

    for i in range(len(flag_array)):
        total_draw_counts += sum(flag_array[i])

    id_list = []
        
    for i in range(len(flag_array)):
        plot_id_data = []
        plot_id_nodata = []
        
        for j in range(len(flag_array[i])):

            normalized_index = color_index / ( total_draw_counts - 1) 
            rgba = colormap(normalized_index) 
            hex_color = '#{:02x}{:02x}{:02x}'.format( int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255) )
 
            if flag_array[i][j] == True:
                if padsinfo[i].hist[j] is not None:
                    ax.plot(
                        padsinfo[i].bincenter[j], 
                        padsinfo[i].hist[j], 
                        label=f'ID:{padsinfo[i].ids[j]}', 
                        color=hex_color, 
                        alpha=0.5,
                        lw=2
                    )

                    plot_id_data.append(padsinfo[i].ids[j])
                    color_index += 1

                else:
                    plot_id_nodata.append(padsinfo[i].ids[j])

        id_list.append([plot_id_data, plot_id_nodata])

    ax.grid(True)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel("Charge [Ch]", fontsize=20)
    ax.set_ylabel("Normalized counts", fontsize=20)

    fig.suptitle(f"Reference : CoBo:{cobo},  AsAd:{asad},  AGET:{aget},  Channel:{channel},", fontsize=22)
    
    if legendflag:
        legend = ax.legend(fontsize=14,ncol=6)
        for line in legend.get_lines():
            line.set_linewidth(5)
                
    plt.tight_layout()

    if showflag:
      plt.show()

    if savepath:
        fig.savefig(savepath)
        if homepageflag:
            filename = os.path.basename(savepath)
            dirname = os.path.dirname(savepath)
            name, ext = os.path.splitext(filename)
            savepath = f"{dirname}/figure002{ext}"
        else:
            savepath = savepath2


    plt.close(fig)

    user_colors=[ ['#B844A0','#36797A'], ['#B844A0','#36797A'], ['#B844A0', '#36797A'] ]
    label_list = [ ['exist (true)', 'no hist (ture)'], ['exist (true)', 'no hist (ture)'], ['exist (true)', 'no hist (ture)'] ]

    util.catmviewer.plot_2d_categories( z_range, x_range, y_range, *padsinfo, *id_list, *label_list, showflag, savepath, user_colors, True)

def draw_charge_1d_histogram_with_geometrical_gate(
        padsinfo=None,
        ref_geometry_id = 1, ref_pad_id = 8, gate_flag = 1, ref_distance = 3, ref_direction=0, 
        showflag=False, savepath=None, legendflag=False, colormapname='rainbow', homepageflag=True, savepath2=None

    ):
     
    if padsinfo is None:
        print("pad array list is none")
        return 
    else:
        if len(padsinfo) != 3:
            print(f"length of pad array list should be 3. current : {len(padsinfo)}")
            return
        
    # plot 
    if not isinstance(ref_pad_id, list):
        ref_position_index = h445util.get_index_from_value(padsinfo[ref_geometry_id].ids, ref_pad_id)
        ref_position = padsinfo[ref_geometry_id].centers[ref_position_index]
        rformatted = " ".join([f"{x:.2f}" for x in ref_position])
        title_text =  f"Reference : Geometry ID = {ref_geometry_id},  Pad ID = {ref_pad_id},  Position (x, y, z) = ({rformatted})"

    else:
        ref_position = []
        rformatted = []
        for i in range(len(ref_pad_id)):
            ref_position_index = h445util.get_index_from_value(padsinfo[ref_geometry_id].ids, ref_pad_id[i]) 
            pos = padsinfo[ref_geometry_id].centers[ref_position_index]
            ref_position.append( pos )
            rformatted.append(f"Ref{i} ID:{ref_pad_id[i]}, Pos:")
            rformatted.append(" ".join([f"{x:.2f}" for x in pos]))
        
        title_text = ""
        for i in range(len(rformatted)):
            title_text += rformatted[i] + " "

    id_list = []
    total_draw_counts = 0

    for i in range(len(padsinfo)):
        plot_id_data = []
        plot_id_nodata = []

        if i is ref_geometry_id:
            for j in range(len(padsinfo[i].ids)):

                distance = 500
                
                if gate_flag == 3:
                    distance = calculate_3d_distance(ref_position, padsinfo[i].centers[j])
                
                elif gate_flag == 1:
                    distance = calculate_1d_distance(ref_position, padsinfo[i].centers[j], ref_direction)

                elif gate_flag == 4:
                    distance0 = calculate_1d_distance(ref_position, padsinfo[i].centers[j], ref_direction[0])
                    distance1 = calculate_1d_distance(ref_position, padsinfo[i].centers[j], ref_direction[1])
                    flag0 = distance0 < ref_direction[2]
                    flag1 = distance1 < ref_direction[3] 
                    distance = flag0 * flag1 if flag0 * flag1 == 1 else 500
                
                elif gate_flag == 5:
                    flags_list = []
                    distance_sum = 0
                    
                    for l in range(len(ref_direction)):
                        flags = []
            
                        for m in range(2):
                            distance_lm = calculate_1d_distance(ref_position[l], padsinfo[i].centers[j], ref_direction[l][m])
                            flags.append( distance_lm < ref_direction[l][m+2] )
                    
                        flags_list.append( flags )

                    for i in range(len(flags_list)):
                        
                        distance_sum += np.prod(flags_list[i])
                    
                    distance = 0 if distance_sum > 0 else 500

                if distance < ref_distance:
                    if padsinfo[i].hist[j] is not None:               
                        plot_id_data.append(padsinfo[i].ids[j])
                        total_draw_counts += 1

                    else:
                        plot_id_nodata.append(padsinfo[i].ids[j])
        
        id_list.append([plot_id_data, plot_id_nodata])
    
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111)

    colormap = getattr(plt.cm, colormapname)
    color_index = 0
    
    for i in range(len(id_list)):
        for j in range(len(id_list[i][0])):
            normalized_index = color_index / (total_draw_counts - 1) 
            rgba = colormap(normalized_index) 
            hex_color = '#{:02x}{:02x}{:02x}'.format( int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255) ) 
            
            ax.plot(
                padsinfo[i].bincenter[id_list[i][0][j]], 
                padsinfo[i].hist[id_list[i][0][j]], 
                label=f'ID:{padsinfo[i].ids[id_list[i][0][j]]}', 
                color=hex_color, 
                alpha=0.5,
                lw=2
            )
            
            color_index += 1

    ax.grid(True)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel("Charge [Ch]", fontsize=20)
    ax.set_ylabel("Normalized counts", fontsize=20)
    fig.suptitle(title_text, fontsize=22)
    
    if legendflag:
        legend = ax.legend(fontsize=14,ncol=6)
        for line in legend.get_lines():
            line.set_linewidth(5)

    plt.tight_layout()

    if showflag:
      plt.show()

    if savepath:
        fig.savefig(savepath)
        if homepageflag:
            filename = os.path.basename(savepath)
            dirname = os.path.dirname(savepath)
            name, ext = os.path.splitext(filename)
            savepath = f"{dirname}/figure002{ext}"
        else:
            savepath = savepath2

    plt.close(fig)

    x_range = (260, -260)
    y_range = (-100, 100)
    z_range = (-270, 220)

    user_colors=[ ['#B844A0','#36797A'], ['#B844A0','#36797A'], ['#B844A0', '#36797A'] ]
    label_list = [ ['exist (true)', 'no hist (ture)'], ['exist (true)', 'no hist (ture)'], ['exist (true)', 'no hist (ture)'] ]

    util.catmviewer.plot_2d_categories( z_range, x_range, y_range, *padsinfo, *id_list, *label_list, showflag, savepath, user_colors, True)

def draw_charge_1d_histogram_with_id_list(
        padsinfo=None, ref_geometry_id=1, ref_pad_id=8, 
        showflag=False, savepath=None, legendflag=False, colormapname='rainbow', homepageflag=True, savepath2=None
    ):

    if padsinfo is None:
        print("pad array list is none")
        return 

    else:
        if len(padsinfo) != 3:
            print(f"length of pad array list should be 3. current : {len(padsinfo)}")
            return

    # plot 
    title_text = 'histogram cheacker'
    total_draw_counts = len(ref_pad_id)
    
    colormap = getattr(plt.cm, colormapname)
    color_index = 0
    
    plot_id_data = []
    plot_id_nodata = []

    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111)

    for i in range(total_draw_counts):

        id_index = h445util.get_index_from_value(padsinfo[ref_geometry_id].ids, ref_pad_id[i])

        if id_index is not None:
            normalized_index = color_index / (total_draw_counts - 1) 
            rgba = colormap(normalized_index) 
            hex_color = '#{:02x}{:02x}{:02x}'.format( int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255) ) 

            if padsinfo[ref_geometry_id].hist[id_index] is not None:
                ax.plot(
                    padsinfo[ref_geometry_id].bincenter[id_index], 
                    padsinfo[ref_geometry_id].hist[id_index], 
                    label=f'ID:{padsinfo[ref_geometry_id].ids[id_index]}', 
                    color=hex_color, 
                    alpha=0.5,
                    lw=2
                )
                plot_id_data.append(ref_pad_id[i])
            else:
                plot_id_nodata.append(ref_pad_id[i])
        
        color_index += 1

    ax.grid(True)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel("Charge [Ch]", fontsize=20)
    ax.set_ylabel("Normalized counts", fontsize=20)
    fig.suptitle(title_text, fontsize=22)

    if legendflag:
        legend = ax.legend(fontsize=14,ncol=6)
        for line in legend.get_lines():
            line.set_linewidth(5)

    plt.tight_layout()

    if showflag:
      plt.show()

    if savepath:
        fig.savefig(savepath)
        if homepageflag:
            filename = os.path.basename(savepath)
            dirname = os.path.dirname(savepath)
            name, ext = os.path.splitext(filename)
            savepath = f"{dirname}/figure002{ext}"
        else:
            savepath = savepath2

    plt.close(fig)
    
    x_range = (260, -260)
    y_range = (-100, 100)
    z_range = (-270, 220)
    
    user_colors=[ ['#B844A0','#36797A'], ['#B844A0','#36797A'], ['#B844A0', '#36797A'] ]
    label_list = [ ['exist (true)', 'no hist (ture)'], ['exist (true)', 'no hist (ture)'], ['exist (true)', 'no hist (ture)'] ]
    id_list = []

    for i in range(len(padsinfo)):
        if i == ref_geometry_id:
            id_list.append([plot_id_data, plot_id_nodata])
        else:
            id_list.append([[],[]])

    util.catmviewer.plot_2d_categories( z_range, x_range, y_range, *padsinfo, *id_list, *label_list, showflag, savepath, user_colors, True)
    
def calculate_scaled_offset(x1, y1, x2, y2, e1, e2, offset_init=0.0, scale_init=1.0):

    def chi2(offset, scale):
        x2_transformed = offset + scale * np.array(x2)
        try:
            interp_func = interp1d(x2_transformed, y2, bounds_error=False, fill_value="extrapolate")
        except Exception as e:
            print("Interpolation error:", e)
            return 1e10 
        y2_interp = interp_func(x1)

        sigma2 = e1 ** 2 + (scale * e2) ** 2
        function = np.sum( (y1 - y2_interp) ** 2 / sigma2 )
        return function 

    m = Minuit(chi2, offset=offset_init, scale=scale_init)
    m.errordef = 1
    m.migrad()

    best_offset = m.values["offset"]
    best_scale = m.values["scale"]
    min_chi2 = m.fval 

    return best_offset, best_scale, min_chi2

def save_all_line(
        padsinfo=None, ref_direction=0, ref_geometry_id=0, 
        gate_flag=0,ref_distance=0, base_path=None, homepageflag=False
    ):

    if padsinfo is None:
        print("pad array list is none")
        return 

    else:
        if len(padsinfo) != 3:
            print(f"length of pad array list should be 3. current : {len(padsinfo)}")
            return
    
    if base_path is None:
        print(f"save path is none")
        return
    
    counts = 0
    direction_dir = 'dum'

    if ref_direction == 0:
        direction_dir='x1d'
        max_loop1 = 2
        max_loop2 = 23
        did = 2024

    elif ref_direction == 2:
        direction_dir='z1d'
        max_loop1 = 88
        max_loop2 = 1
        did = 23

    for j in range(max_loop1):
        for i in range(max_loop2):

            num = i + j * did

            save_fig_path_hist = pathlib.Path(f"{base_path}/figs/python/pulse_shape_chrage/{direction_dir}/hist/{num}.png")
            save_fig_path_pad  = pathlib.Path(f"{base_path}/figs/python/pulse_shape_chrage/{direction_dir}/pad/{num}.png")

            base_save_path = f"{base_path}/figs/python/pulse_shape_chrage"

            if counts == 0:
                print(f"base output path : {base_save_path}")
                save_fig_path_hist.parent.mkdir(parents=True, exist_ok=True)
                save_fig_path_pad.parent.mkdir(parents=True, exist_ok=True)

            print(f"path : {save_fig_path_hist}")

            draw_charge_1d_histogram_with_geometrical_gate(
                padsinfo,
                ref_geometry_id=ref_geometry_id,  
                ref_pad_id=num, 
                gate_flag=gate_flag, 
                ref_distance=ref_distance,
                ref_direction=ref_direction,
                homepageflag=homepageflag,
                savepath=save_fig_path_hist,
                savepath2=save_fig_path_pad
            )

            counts += 1

def save_all_chip(
        padsinfo=None, base_path=None, homepageflag=False
    ):

    if padsinfo is None:
        print("pad array list is none")
        return 

    else:
        if len(padsinfo) != 3:
            print(f"length of pad array list should be 3. current : {len(padsinfo)}")
            return
    
    if base_path is None:
        print(f"save path is none")
        return
    
    counts = 0
    
    for i in range(4):
        for j in range(4):
            for k in range(4):

                save_fig_path_hist = pathlib.Path(f"{base_path}/figs/python/pulse_shape_chrage/get/hist/cobo{i}asad{j}aget{k}.png")
                save_fig_path_pad  = pathlib.Path(f"{base_path}/figs/python/pulse_shape_chrage/get/pad/cobo{i}asad{j}aget{k}.png")
                base_save_path = f"{base_path}/figs/python/pulse_shape_chrage"

                if counts == 0:
                    print(f"base output path : {base_save_path}")
                    save_fig_path_hist.parent.mkdir(parents=True, exist_ok=True)
                    save_fig_path_pad.parent.mkdir(parents=True, exist_ok=True)
               
                print(f"path : {save_fig_path_hist}")

                draw_charge_1d_histogram_with_getdevice(
                    cobo=i, asad=j, aget=k, channel=-1, 
                    homepageflag=homepageflag,
                    savepath=save_fig_path_hist,
                    savepath2=save_fig_path_pad
                )

            counts += 1

def execute_relative_calibration(padsinfo=None, base_path=None, save_path=None):

    if padsinfo is None:
        print("pad array list is none")
        return 

    else:
        if len(padsinfo) != 3:
            print(f"length of pad array list should be 3. current : {len(padsinfo)}")
            return

    input_path = f"{base_path}/data/kr80/padcalib/catm-map-get-readoutpad.pkl"

    print(input_path)

    ref_data = h445util.load_class_object(input_path)    

    threshold1 = h445util.classify_indices(ref_data.global_threshold)
    threshold2 = h445util.classify_indices(ref_data.LSB_threshold)

    print("key list : global_threshold =", threshold1.keys(),", LSB_threshold =", threshold2.keys())

    list1 = list( threshold1.values() )
    list2 = list( threshold2.values() )

    key1 = list( threshold1.keys() )
    key2 = list( threshold2.keys() )

    commons = []
    labels = []

    for i in range(len(list1)):
        for j in range(len(list2)):
            common = list(set(list1[i]) & set(list2[j]))
            commons.append(common)
            labels.append(f"threshold : ({key1[i]}, {key2[j]})")

    fit_results=[]

    ref_ids = [1046, 1046, 1046]
    print(ref_ids)
    for j in range(len(ref_ids)):
        ref_y  = np.array(padsinfo[1].hist[ref_ids[j]])
        ref_x  = np.array(padsinfo[1].bincenter[ref_ids[j]])

        replaced_e = [1 if i == 0 else i for i in padsinfo[1].error[ref_ids[j]]]
        ref_e  = np.array(replaced_e)

        for i in range(len(commons[j])):
            data_id = padsinfo[1].ids[commons[j][i]]
           
            best_p0 = 1000#"unsigned_p0"
            best_p1 = 2000#"unsigned_p1"
            best_chi2 = -1000

            if padsinfo[1].hist[commons[j][i]] is not None:
                data_y = np.array(padsinfo[1].hist[commons[j][i]])
                data_x = np.array(padsinfo[1].bincenter[commons[j][i]])

                errors = padsinfo[1].error[commons[j][i]]
                replaced_e = [1 if err == 0 else err for err in errors]
                data_e  = np.array(replaced_e)

                best_p0, best_p1, best_chi2 = calculate_scaled_offset(ref_x, ref_y, data_x, data_y, ref_e, data_e, offset_init=0.0, scale_init=1.0)

            cobo = padsinfo[1].cobos[commons[j][i]]
            asad = padsinfo[1].asads[commons[j][i]]
            aget = padsinfo[1].agets[commons[j][i]]
            channel = padsinfo[1].channels[commons[j][i]]
            integrated_hist = padsinfo[1].hist_sum[commons[j][i]]
            id = padsinfo[1].ids[commons[j][i]]
            
            fit_results.append([id,best_p0,best_p1,best_chi2,cobo,asad,aget,channel,integrated_hist])

        calibration_file_path = f"{base_path}/data/kr80/padcalib/pad_caliburation_electronics.txt"
        df_calib = pd.DataFrame(fit_results)
        df_calib.to_csv(calibration_file_path, index=False, sep=" ")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-rpi", "--ref-pad-id", nargs="*", type=int, help='a list of reference id', default=8)

    parser.add_argument("-t","--type", help="select method", type=int, default=0)
    parser.add_argument("-rco", "--ref-cobo", help="select reference cobo id [-1:all]", type=int, default=-1)
    parser.add_argument("-ras", "--ref-asad", help="select reference asad id [-1:all]", type=int, default=-1)
    parser.add_argument("-rag", "--ref-aget", help="select reference aget id [-1:all]", type=int, default=-1)
    parser.add_argument("-rch", "--ref-channel", help="select reference channel id [-1:all]", type=int, default=-1)

    parser.add_argument("-rgi", "--ref-geometry-id", help="select reference cobo id [-1:all]", type=int, default=1)
    parser.add_argument("-rgf", "--ref-gate-flag", help="select reference aget id [-1:all]", type=int, default=1)
    parser.add_argument("-rdv", "--ref-distance-value", help="select reference channel id [-1:all]", type=int, default=3)
    parser.add_argument("-rav", "--ref-axis-value", help="select reference channel id [-1:all]", type=int, default=0)

    parser.add_argument("-cm","--colormap-name", help="select colormap", type=str, default="rainbow")

    parser.add_argument("-pf", "--plot-flag", help="flag of plot drawing", action="store_true")
    parser.add_argument("-sf", "--save-flag", help="flag of plot saving", action="store_true")
    parser.add_argument("-lf", "--legend-flag", help="flag of plot legend", action="store_true")

    args = parser.parse_args()

    check_type: int = args.type
    colormap_name: str = args.colormap_name
   
    plot_flag: bool = args.plot_flag
    save_flag: bool = args.save_flag
    legend_flag: bool = args.legend_flag

    ref_geometry_id = args.ref_pad_id 
    cobo = args.ref_cobo
    asad = args.ref_asad
    aget = args.ref_aget
    channel = args.ref_channel
    ref_geometry_id = args.ref_geometry_id
    ref_pad_id = args.ref_pad_id
    gate_flag = args.ref_gate_flag
    ref_distance = args.ref_distance_value
    ref_direction = args.ref_axis_value

    maps_path, base_path, homepage_path = h445util.load_maps_path()

    input_histograms =  'data/kr80/padcalib/normalized_charge_id.csv'
    histogram_path = f"{base_path}/{input_histograms}" 

    padsinfo  = h445util.get_padarrays_with_map(*maps_path)
    padsinfo = merge_padarray_and_histogram(padsinfo, histogram_path)
    save_path = homepage_path if save_flag else None

    if check_type == 0:        
        draw_charge_1d_histogram_with_getdevice(
            padsinfo=padsinfo, cobo=cobo,asad=asad, aget=aget, channel=channel,
            showflag=plot_flag, savepath=save_path, legendflag=legend_flag,
            colormapname=colormap_name, homepageflag=True, savepath2=None
        )

    elif check_type == 1:
        draw_charge_1d_histogram_with_geometrical_gate(
            padsinfo=padsinfo, ref_geometry_id=ref_geometry_id, ref_pad_id=ref_pad_id, 
            gate_flag=gate_flag, ref_distance=ref_distance, ref_direction=ref_direction,
            showflag=plot_flag, savepath=save_path, legendflag=legend_flag,
            colormapname=colormap_name, homepageflag=True, savepath2=None
        )

    elif check_type == 2: 
        save_all_line(
            padsinfo=padsinfo, 
            ref_direction=ref_direction, ref_geometry_id=ref_geometry_id,
            gate_flag=gate_flag,ref_distance=ref_distance, base_path=base_path,
            homepageflag=False
        )

    elif check_type == 3: 
        save_all_chip( padsinfo=padsinfo, base_path=base_path, homepageflag=False)
    
    elif check_type == 4:
        execute_relative_calibration(padsinfo ,base_path, save_path) 

if __name__ == "__main__":
    main()
