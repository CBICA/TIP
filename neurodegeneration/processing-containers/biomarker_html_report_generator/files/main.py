import os
import shutil
import pandas as pd

from roi_quantifier.roi_quantifier import roi_quantifier_main
from spare_calculator.spare_calculator import spare_main
from normative_biomarker_visualizer.normative_biomarker_visualizer import biomarker_main
from brainvisualize.vtkBrainVisual import _main as brainvisual_main
from csv_extraction.csv_extraction import _main as csv_main
from html_generator.html_generator import _main as html_main

def main(muse_roi, dlicv_mask, wmls_mask, dcm_json):
    
    ## Create a tmp directory to store the intermediate results
    tmp_dir = '/tmp_folder'
    
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        
    os.mkdir(tmp_dir)
    
    print('tmp dir: ', tmp_dir)
    
    tmp_file_path = tmp_dir + '/tmp.pdf'
    print('tmp pdf: ', tmp_file_path)
    #exit(1)
    
    ######################## Put all the intermediate output into the out folder##########################
    #      [UID]_all_MuseROIs_name.pkl – A dictionary with {‘ROI Name’: ‘ROI volume’}                    #
    #      [UID]_all_MuseROIs_num.pkl - A dictionary with {‘ROI Index’: ‘ROI volume’}                    #
    #      [UID]_allz_num.pkl - A dictionary with {‘ROI Index’: ‘Normalized ROI volume’}                 #
    #      [UID]_allz.pkl - A dictionary with {‘ROI Name’: ‘Normalized ROI volume’}                      #
    #      [UID]_dfPat.pkl - Patient Information includes patient ID, Age, Sex, dates                    #
    #      [UID]_dfRef.pkl - Reference Information                                                       #   
    #      [UID]_dfSub.pkl - Subject Information                                                         #
    #      [UID]_WMLSref.pkl - WMLS Reference Information                                                #
    ######################################################################################################
    
    dfSub, dfRef, WMLSref, dfPat, allz_num, allz, all_MuseROIs_num, all_MuseROIs_name, MuseROI = roi_quantifier_main(muse_roi, dlicv_mask, wmls_mask, dcm_json, tmp_file_path)
    print('ROI QUANTIFIER DONE!')
    ######################## Put all the intermediate output into the tmp folder##########################
    #      [UID]_spareAD.pkl – SPARE AD score (single value)                                             #
    #      [UID]_spareBA.pkl - SPARE BA score (single value)                                             #
    ######################################################################################################
    
    spareAD, spareBA = spare_main(dfSub, MuseROI, tmp_file_path)
    print('SPARE DONE !')
    ######################## Put all the intermediate output into the out folder##########################
    #      [UID]_flagtable.pkl –  html file contains table information (Brain Volumetry and Comparison   #
    #                                  with Normative Harmonized Population Values)                      #                      
    #      [UID]_roisubsettable.pkl - A html file contains table information (Regions that differ the    #
    #                                 most from normal                                                   #
    #      [UID]_plot.png                                                                                #
    #      [UID]_all_MuseROIs_num.pkl - A dictionary with {‘ROI Index’: ‘ROI volume’}                    #
    #      [UID]_all_MuseROIs_name.pkl – A dictionary with {‘ROI Name’: ‘ROI volume’}                    #
    #      [UID]_all_MuseROIs_num.pkl - A dictionary with {‘ROI Index’: ‘ROI volume’}                    #
    #      [UID]_allz_num.pkl - A dictionary with {‘ROI Index’: ‘Normalized ROI volume’}                 #
    #      [UID]_allz.pkl - A dictionary with {‘ROI Name’: ‘Normalized ROI volume’}                      #
    #      [UID]_dfPat.pkl - Patient Information includes patient ID, Age, Sex, dates                    #
    #      [UID]_dfRef.pkl - Reference Information                                                       #   
    #      [UID]_dfSub.pkl - Subject Information                                                         #
    #      [UID]_WMLSref.pkl - WMLS Reference Information                                                #
    ######################################################################################################
    
    biomarker_main(dfSub, dfRef, WMLSref, allz_num, allz, all_MuseROIs_name, spareAD, spareBA, tmp_file_path)
    print('BOIMARKER DONE !')
    ############################ Brain Visualize ########################################################
    
    #brainvisual_main(muse_roi[0], allz_num, tmp_file_path)
    brainvisual_main('../refs/20181205_124179_T1_LPS_muse_relabeled.nii.gz', allz_num, tmp_file_path)

    print('BRAIN DONE !')

    ############################ CVS Extraction #########################################################
    
    csv_main(dfSub, dfRef, WMLSref, allz_num, allz, all_MuseROIs_name, spareAD, spareBA, tmp_file_path)
    print('CSV DONE !')
    ############################ html extraction ########################################################
    
    dfPat = pd.read_pickle(tmp_dir + '/tmp_dfPat.pkl')
    table = pd.read_pickle(tmp_dir + '/tmp_roisubsettable.pkl')
    flagtable = pd.read_pickle(tmp_dir + '/tmp_flagtable.pkl')

    #absolute_path = os.path.abspath(os.getcwd())

    #pdf_path = absolute_path + tmp_file_path
    #print('pdf_path : ', pdf_path)
    html_main(tmp_file_path, dfPat, table, flagtable)
    
    print('HTML DONE !')

    #output is a single html file + all files required by csv_extraction container
