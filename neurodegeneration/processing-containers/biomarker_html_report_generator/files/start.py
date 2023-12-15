import os
import glob
import shutil
from main import main as main

# For local testng
os.environ["WORKFLOW_DIR"] = "/sharedFolder/data/F8_F12_M2_M3_M5/M17"
os.environ["BATCH_NAME"] = "batch"
os.environ["OPERATOR_OUT_DIR"] = "/sharedFolder/data/F8_F12_M2_M3_M5/M17/out"
os.environ["OPERATOR_IN_DCM_METADATA_DIR"] = "GetT1Metadata"
os.environ["OPERATOR_IN_DIR_ICV"] = "extract_dlicv_result"
os.environ["OPERATOR_IN_DIR_ROI"] = "extract_muse_lps_result"
os.environ["OPERATOR_IN_DIR_WMLS"] = 'wmls-output'

# From the template
batch_folders = sorted([f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))])

for batch_element_dir in batch_folders:
    icv = []
    roi = []
    wmls = []

    print(f'Checking for nii/json files')

    if "None" not in os.environ["OPERATOR_IN_DIR_ICV"]:
        print("icv folder provided")
        icv_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_ICV'])
        icv = sorted(glob.glob(os.path.join(icv_input_dir, "*.nii*"), recursive=True))

    if "None" not in os.environ["OPERATOR_IN_DIR_ROI"]:
        print("roi folder provided")
        roi_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_ROI'])
        roi = sorted(glob.glob(os.path.join(roi_input_dir, "*.nii*"), recursive=True))

    if "None" not in os.environ["OPERATOR_IN_DIR_WMLS"]:
        print("wmls folder provided")
        wmls_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_WMLS'])
        wmls = sorted(glob.glob(os.path.join(wmls_input_dir, "*.nii*"), recursive=True))

    tmp_json = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DCM_METADATA_DIR'])

    element_output_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
    if not os.path.exists(element_output_dir):
        os.makedirs(element_output_dir)
    
    # The processing algorithm
    json_file = sorted(glob.glob(os.path.join(tmp_json, "*.json*"), recursive=True))
    
    # Throws error if ALL components are missing - may want to change in future
    if len(icv) == 0 and len(roi) == 0 and len(wmls) == 0:
        print("No nifti file(s) found!")
        exit(1)
    if len(json_file) != 1:
        print("Incorrect # of JSON file in directory")
        exit(1)
    else:
        json_file = json_file[0]
        if not os.path.exists(element_output_dir):
            os.makedirs(element_output_dir)

        ### The output file path: pdf file contains the final report and csv file contains patient information
        pdf_file_path = os.path.join(element_output_dir, "{}.pdf".format(os.path.basename(batch_element_dir)))
        csv_path = os.path.join(element_output_dir, "{}_info.csv".format(os.path.basename(batch_element_dir)))

        ### Created the temporal folder to store intermediate results
        main(roi, icv, wmls, json_file) 
        
        ### Move the intermeidate results to the final folder
        shutil.move('/tmp_folder/tmp.pdf', pdf_file_path)
        shutil.move('/tmp_folder/tmp_info.csv', csv_path)