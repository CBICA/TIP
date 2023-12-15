import pandas as pd
import numpy as np
import pickle
import json
import nibabel as nib
from datetime import datetime
import csv as _csv
import os as _os
import copy

#### Hardcoded reference data ###
# Harmonized reference MUSE values #
MUSE_Ref_Values = '/refs/combinedharmonized_out.csv'
# Left and Right ROIs indices as well as ROI name to ROI number equivalency #
maphemi = pd.read_csv('/refs/MUSE_ROI_Dictionary.csv')
# Single ROI to combined ROI mapping #
MUSE_ROI_Mapping = '/refs/MUSE_DerivedROIs_Mappings.csv'
# Harmonized reference WMLS values #
WMLS_Ref_Values = '/refs/WMLS_combinedrefs.csv'

################################################ FUNCTIONS ################################################

##########################################################################
### Function to read demographic info from json
def readSubjInfo(subjfile):
	
	if subjfile.endswith('.csv'):
		subjAge = None
		subjSex = None
		if subjfile is not None:
			d={}
			with open(subjfile) as f:
				reader = _csv.reader( f )
				subDict = {rows[0]:rows[1] for rows in reader}
	
	if subjfile.endswith('.json'):
		with open(subjfile, 'r') as read_file:
			data = json.load(read_file)
			
			## WARNING: Trying to parse fields in a robust way
			## This part is based on assumptions on dicom field names
			## it should be tested carefully and updated for different patterns
			subID = [value for key, value in data.items() if 'PatientID' in key][0]
			
			for key, value in data.items():
				if ('PatientAge' in key) and ('Y' in str(value)):
					print("Successfully identified patient age in JSON...")
					subAge = float(str(value).replace('Y',''))
					break

			subSex = [value for key, value in data.items() if 'PatientSex' in key][0]
			subExamDate = [value for key, value in data.items() if 'StudyDate_date' in key][0]
			subExamDate = datetime.strptime(subExamDate, "%Y-%m-%d").strftime("%m/%d/%Y")
			
			subDict = {'MRID':subID, 'Age':subAge, 'Sex':subSex, "ExamDate":subExamDate}

	return subDict

### Function to calculate any mask volume
def calcMaskVolume(maskfile):
	
	### Check input mask
	if not maskfile:
		raise Exception('Input file not provided!!!')
		
	### Read the input image
	roinii = nib.load(maskfile)
	roiimg = roinii.get_fdata()

	### Get voxel dimensions
	voxdims1, voxdims2, voxdims3 = roinii.header.get_zooms()
	voxvol = float(voxdims1)*float(voxdims2)*float(voxdims3)

	### Calculate mask volume
	maskVol = voxvol * np.sum(roiimg.flatten()>0)
	
	return maskVol

### Function to calculate derived ROI volumes for MUSE
def calcRoiVolumes(maskfile, mapcsv, dfRef, subDict):
	
	### Check input mask
	if not maskfile:
		raise Exception('ERROR: Input file not provided!!!')

	### Read the input image
	roinii = nib.load(maskfile)
	roiimg = roinii.get_fdata()

	### Get voxel dimensions
	voxdims1, voxdims2, voxdims3 = roinii.header.get_zooms()
	voxvol = float(voxdims1)*float(voxdims2)*float(voxdims3)

	### Calculate ROI count and volume
	ROIs, Counts = np.unique(roiimg, return_counts=True)
	ROIs = ROIs.astype(int)
	Volumes = voxvol * Counts

	### Create an array indexed from 0 to max ROI index
	###   This array will speed up calculations for calculating derived ROIs
	###   Instead of adding ROIs in a loop, they are added at once using: 
	###          np.sum(all indexes for a derived ROI)  (see below)
	VolumesInd = np.zeros(ROIs.max()+1)
	VolumesInd[ROIs] = Volumes
	
	### Calculate derived volumes
	DerivedROIs = []
	DerivedVols = []
	ROIlist = []
	Vollist = []
	with open(mapcsv) as mapcsvfile:
		reader = _csv.reader(mapcsvfile, delimiter=',')
		# Read each line in the csv map files
		for row in reader:
			row = list(filter(lambda a: a != '', row))
			# Append the ROI number to the list
			DerivedROIs.append(row[0])
			roiInds = [int(x) for x in row[2:]]
			DerivedVols.append(np.sum(VolumesInd[roiInds]))

	### Declare objects we want to store
	all_MuseROIs =  dict(zip(DerivedROIs, DerivedVols))
	dictr = {}
	allz = {}
	allz_num = {}
	all_MuseROIs_name = {}

	# Get all left and right ROIs
	Rinds = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'R'), 'ROI_INDEX'].values)
	Linds = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'L'), 'ROI_INDEX'].values)

	#### Correct ref values temporarily for age + gender controlled z-score calculation ####
	nonROI = list(dfRef.columns[dfRef.columns.str.contains('AI')].values)
	nonROI.extend(['MRID','Study','PTID','Age','Sex','Diagnosis_nearest_2.0','SITE','Date','ICV','SPARE_AD','SPARE_BA'])
	dfRefTmp = dfRef.copy(deep=True)
	# Rename 702 to ICV
	dfRefTmp['ICV'] = dfRefTmp['702']
 
	# Do the ICV-adjust
	dfRefTmp[dfRefTmp.columns.difference(nonROI)] = dfRefTmp.apply(lambda row: row[dfRefTmp.columns.difference(nonROI)].div(row['ICV']).mul(dfRefTmp[ (dfRefTmp['Age'] >= row['Age'] -3 ) & ## Ref age >= age - 3
                                                                                                                                           (dfRefTmp['Age'] <= row['Age'] +3 ) & ## Ref age <= age + 3
                                                                                                                                           (dfRefTmp['Diagnosis_nearest_2.0'] == 'CN')]['ICV'].mean()), axis = 1) ## CN Only
	# Convert from mm^3 to cm^3
	dfRefTmp[dfRefTmp.columns.difference(nonROI)] = dfRefTmp[dfRefTmp.columns.difference(nonROI)]/1000

	## Create anatomical structure to z-score equivalency for the subject; only considering ROIs Ilya said to use! - z-score is calculated across all same sex ref patients
	for i in list(all_MuseROIs.keys()):
		# Only look at single ROIs
		if int(i) <= 207:
			allz[i] = ((all_MuseROIs[i]/1000) - dfRefTmp[i].mean())/(dfRefTmp[i].std())
			allz_num[i] = allz[i]
			name = list(maphemi.loc[(maphemi['ROI_INDEX'] == int(i)), 'ROI_NAME'].values)[0]
			allz[name] = allz.pop(i)
			all_MuseROIs_name[name] = all_MuseROIs[i]
		else:
			continue
	
	### Select patient ROIs to report and add colloquial terminology to selected patient (dictr) and reference ROIs (dfRef) ###
	### Brain volumes ###
	# Total brain volume
	dictr["Total Brain Volume"] = all_MuseROIs["701"]
	
	#dfRef.rename(columns={"701":"Total Brain Volume"}, inplace = True)
	dfRef['Total Brain Volume'] = dfRef['701']

	# Right brain volume (1/2 of all indivisible ROIs: 4,11,35,46,71,72,73,95)
	dictr["Right Brain Volume"] = np.sum(VolumesInd[Rinds]) + np.sum(VolumesInd[[4,11,35,71,72,73,95]])/2
	dfRef["Right Brain Volume"] = dfRef[[str(i) for i in Rinds]].sum(axis=1) + (dfRef[['4','11','35','71','72','73','95']].sum(axis=1))/2
	# Left brain volume (1/2 of all indivisible ROIs: 4,11,35,46,71,72,73,95)
	dictr["Left Brain Volume"] = np.sum(VolumesInd[Linds]) + np.sum(VolumesInd[[4,11,35,71,72,73,95]])/2
	dfRef["Left Brain Volume"] = dfRef[[str(i) for i in Linds]].sum(axis=1) + (dfRef[['4','11','35','71','72','73','95']].sum(axis=1))/2
	# Asymmetry index for brain
	dictr["Brain AI"] = abs(dictr["Left Brain Volume"] - dictr["Right Brain Volume"])/((dictr["Left Brain Volume"] + dictr["Right Brain Volume"])/2)
	dfRef["Brain AI"] = (dfRef["Left Brain Volume"] - dfRef["Right Brain Volume"]).abs().div((dfRef["Left Brain Volume"] + dfRef["Right Brain Volume"])/2,axis=0)

	### Brainstem ### - had to adjust this ROI
	dictr["Total Brainstem Volume"] = all_MuseROIs["35"] + all_MuseROIs["61"] + all_MuseROIs["62"]
	# dfRef["35"] = dfRef["35"] + dfRef["61"] + dfRef["62"]
	# dfRef.rename(columns={"35":"Total Brainstem Volume"}, inplace = True)
	dfRef['Total Brainstem Volume'] = dfRef["35"] + dfRef["61"] + dfRef["62"]

	### Ventricles###
	# Total ventricle
	dictr["Total Ventricle Volume"] = all_MuseROIs["509"]
	
 
	#dfRef.rename(columns={"509":"Total Ventricle Volume"}, inplace = True)
	dfRef['Total Ventricle Volume'] = dfRef['509']

	# Right ventricle (1/2 of 3rd + 4th)
	dictr["Right Ventricle Volume"] = all_MuseROIs["49"] + all_MuseROIs["51"] + all_MuseROIs["4"]/2 + all_MuseROIs["11"]/2
	dfRef["Right Ventricle Volume"] = dfRef["49"] + dfRef["51"] + dfRef["4"]/2 + dfRef["11"]/2
	# Left ventricle (1/2 of 3rd + 4th)
	dictr["Left Ventricle Volume"] = all_MuseROIs["50"] + all_MuseROIs["52"] + all_MuseROIs["4"]/2 + all_MuseROIs["11"]/2
	dfRef["Left Ventricle Volume"] = dfRef["50"] + dfRef["52"] + dfRef["4"]/2 + dfRef["11"]/2
	# Asymmetry index for ventricles
	dictr["Ventricle AI"] = abs(dictr["Left Ventricle Volume"] - dictr["Right Ventricle Volume"])/((dictr["Left Ventricle Volume"] + dictr["Right Ventricle Volume"])/2)
	dfRef["Ventricle AI"] = (dfRef["Left Ventricle Volume"] - dfRef["Right Ventricle Volume"]).abs().div((dfRef["Left Ventricle Volume"] + dfRef["Right Ventricle Volume"])/2,axis=0)

	### Cerebellum ###
	# Total cerebellum
	dictr["Total Cerebellum Volume"] = all_MuseROIs["502"]
	
	#dfRef.rename(columns={"502":"Total Cerebellum Volume"}, inplace = True)
	dfRef['Total Cerebellum Volume'] = dfRef['502']
 
	# Right cerebellum
	dictr["Right Cerebellum Volume"] = all_MuseROIs["518"]
	dfRef["Right Cerebellum Volume"] = dfRef["518"]
	# Left cerebellum
	dictr["Left Cerebellum Volume"] = all_MuseROIs["510"]
	dfRef["Left Cerebellum Volume"] = dfRef["510"]
	# Asymmetry index for cerebellum
	dictr["Cerebellum AI"] = abs(dictr["Left Cerebellum Volume"] - dictr["Right Cerebellum Volume"])/((dictr["Left Cerebellum Volume"] + dictr["Right Cerebellum Volume"])/2)
	dfRef["Cerebellum AI"] = (dfRef["Left Cerebellum Volume"] - dfRef["Right Cerebellum Volume"]).abs().div((dfRef["Left Cerebellum Volume"] + dfRef["Right Cerebellum Volume"])/2,axis=0)

	### Gray Matter ###
	# Total gray matter
	dictr["Total Gray Matter Volume"] = all_MuseROIs["601"]
	dfRef["601"] = dfRef["601"]
	#dfRef.rename(columns={"601":"Total Gray Matter Volume"}, inplace = True)
	dfRef['Total Gray Matter Volume'] = dfRef['601']
 
	# Right gray matter - had to adjust this ROI with 1/2 of certain ROIs
	dictr["Right Gray Matter Volume"] = all_MuseROIs["613"] + all_MuseROIs["71"]/2 + all_MuseROIs["72"]/2 + all_MuseROIs["73"]/2
	# dfRef["613"] = dfRef["613"] + dfRef["71"]/2 + dfRef["72"]/2 + dfRef["73"]/2
	# dfRef.rename(columns={"613":"Right Gray Matter Volume"}, inplace = True)
 
	dfRef["Right Gray Matter Volume"] = dfRef["613"] + dfRef["71"]/2 + dfRef["72"]/2 + dfRef["73"]/2
 
	# Left gray matter - had to adjust this ROI with 1/2 of certain ROIs
	dictr["Left Gray Matter Volume"] = all_MuseROIs["606"] + all_MuseROIs["71"]/2 + all_MuseROIs["72"]/2 + all_MuseROIs["73"]/2
	# dfRef["606"] = dfRef["606"] + dfRef["71"]/2 + dfRef["72"]/2 + dfRef["73"]/2
	# dfRef.rename(columns={"606":"Left Gray Matter Volume"}, inplace = True)
	dfRef['Left Gray Matter Volume'] =  dfRef["606"] + dfRef["71"]/2 + dfRef["72"]/2 + dfRef["73"]/2

	# Asymmetry index for gray matter
	dictr["Gray Matter AI"] = abs(dictr["Left Gray Matter Volume"] - dictr["Right Gray Matter Volume"])/((dictr["Left Gray Matter Volume"] + dictr["Right Gray Matter Volume"])/2)
	dfRef["Gray Matter AI"] = (dfRef["Left Gray Matter Volume"] - dfRef["Right Gray Matter Volume"]).abs().div((dfRef["Left Gray Matter Volume"] + dfRef["Right Gray Matter Volume"])/2,axis=0)

	### White Matter ###
	# Total white matter
	dictr["Total White Matter Volume"] = all_MuseROIs["604"]
	#dfRef.rename(columns={"604":"Total White Matter Volume"}, inplace = True)
	dfRef['Total White Matter Volume'] = dfRef['604']

	# Right white matter - had to adjust this ROI with 1/2 of certain ROIs
	dictr["Right White Matter Volume"] = all_MuseROIs["614"] + all_MuseROIs["95"]/2
	#dfRef.rename(columns={"614":"Right White Matter Volume"}, inplace = True)
	dfRef['Right White Matter Volume'] = dfRef['614']
 
	# Left white matter - had to adjust this ROI with 1/2 of certain ROIs
	dictr["Left White Matter Volume"] = all_MuseROIs["607"] + all_MuseROIs["95"]/2
	#dfRef.rename(columns={"607":"Left White Matter Volume"}, inplace = True)
	dfRef['Left White Matter Volume'] = dfRef['607']
 
	# Asymmetry index for white matter
	dictr["White Matter AI"] = abs(dictr["Left White Matter Volume"] - dictr["Right White Matter Volume"])/((dictr["Left White Matter Volume"] + dictr["Right White Matter Volume"])/2)
	dfRef["White Matter AI"] = (dfRef["Left White Matter Volume"] - dfRef["Right White Matter Volume"]).abs().div((dfRef["Left White Matter Volume"] + dfRef["Right White Matter Volume"])/2, axis = 0)

	### Hippocampus ###
	# Total Hippocampus
	dictr["Total Hippocampus Volume"] = all_MuseROIs["47"] + all_MuseROIs["48"]
	dfRef["Total Hippocampus Volume"] = dfRef["47"] + dfRef["48"]
	# Right Hippocampus
	dictr["Right Hippocampus Volume"] = all_MuseROIs["47"]
	dfRef["Right Hippocampus Volume"] = dfRef["47"]
	#Left Hippocampus
	dictr["Left Hippocampus Volume"] = all_MuseROIs["48"]
	dfRef["Left Hippocampus Volume"] = dfRef["48"]
	# Asymmetry index for hippocampus
	dictr["Hippocampus AI"] = abs(dictr["Left Hippocampus Volume"] - dictr["Right Hippocampus Volume"])/((dictr["Left Hippocampus Volume"] + dictr["Right Hippocampus Volume"])/2)
	dfRef["Hippocampus AI"] = (dfRef["Left Hippocampus Volume"] - dfRef["Right Hippocampus Volume"]).abs().div((dfRef["Left Hippocampus Volume"] + dfRef["Right Hippocampus Volume"])/2, axis = 0)

	return dictr, dfRef, allz, allz_num, all_MuseROIs, all_MuseROIs_name

### Function adjust all ROIs by ICV and scales down by 1000 ###
def ICVAdjust(dfSub, dfRef,WMLSref,all_MuseROIs_num,all_MuseROIs_name):

	## ICV-adjustment for subject values
	nonROI = list(dfSub.columns[dfSub.columns.str.contains('AI')].values)
	othervars = ['MRID','Age','Sex','ICV']
	nonROI.extend(othervars)

	#### ICV-adjustment for subject values ####
	const_factor = dfRef[(dfRef['Age'] >= dfSub.loc[0, 'Age'] - 3) & 
                         (dfRef['Age'] <= dfSub.loc[0, 'Age'] + 3) & 
                         (dfRef['Diagnosis_nearest_2.0'] == 'CN')]['ICV'].mean() /1000
	
	dfSub[dfSub.columns.difference(nonROI)] = dfSub[dfSub.columns.difference(nonROI)].div(dfSub['ICV'], axis=0)* const_factor

	all_MuseROIs_num = {key: ((value / dfSub['ICV'].values[0])*const_factor)for key, value in all_MuseROIs_num.items()}
	all_MuseROIs_name = {key: ((value / dfSub['ICV'].values[0])*const_factor) for key, value in all_MuseROIs_name.items()}
 
    # make 702 into actual ICV value
	all_MuseROIs_num['702'] = dfSub['ICV'].values[0]
 
     
    ## ICV-adjustment for ROI reference values
	nonROI = list(dfRef.columns[dfRef.columns.str.contains('AI')].values)
	othervars = ['MRID','Study','PTID','Age','Sex','Diagnosis_nearest_2.0','SITE','Date','ICV','SPARE_AD','SPARE_BA']
	nonROI.extend(othervars)
 
	dfRef[dfRef.columns.difference(nonROI)] = dfRef.apply(lambda row: row[dfRef.columns.difference(nonROI)].div(row['ICV']).mul(dfRef[ (dfRef['Age'] >= row['Age'] -3 ) & ## Ref age >= age - 3
                                                                                                                                       (dfRef['Age'] <= row['Age'] +3 ) & ## Ref age <= age + 3
                                                                                                                                       (dfRef['Diagnosis_nearest_2.0'] == 'CN')]['ICV'].mean()), axis = 1) ## CN Only
	dfRef[dfRef.columns.difference(nonROI)] = dfRef[dfRef.columns.difference(nonROI)]/1000
	


	## ICV-adjustment for WMLS reference values
	nonROI = ['ID','Phase','PTID','Age','Sex','Diagnosis_nearest_2.0','Date','ICV']

	WMLSref[WMLSref.columns.difference(nonROI)]  = WMLSref.apply(lambda row: row[WMLSref.columns.difference(nonROI)].div(row['ICV']).mul(WMLSref[ (WMLSref['Age'] >= row['Age'] -3 ) & ## Ref age >= age - 3
                                                                                                                                                  (WMLSref['Age'] <= row['Age'] +3 ) & ## Ref age <= age + 3
                                                                                                                                                  (WMLSref['Diagnosis_nearest_2.0'] == 'CN')]['ICV'].mean()), axis = 1) ## CN Only
	WMLSref[WMLSref.columns.difference(nonROI)] = WMLSref[WMLSref.columns.difference(nonROI)]/1000


	return dfSub, dfRef, WMLSref, all_MuseROIs_num, all_MuseROIs_name



############## MAIN ##############
def roi_quantifier_main(roi, icv, wmls, _json, out_path):
	UID = _os.path.basename(out_path.removesuffix(".pdf"))
 
	##### Read and initial filter for reference ROI values###################
	dfRef = pd.read_csv(MUSE_Ref_Values).dropna()
	out = _os.path.dirname(out_path)
	dfRef['Date'] = pd.to_datetime(dfRef.Date)
	dfRef = dfRef.sort_values(by='Date')
	# Get first-time points only from reference values
	dfRef = dfRef.drop_duplicates(subset=['PTID'], keep='first')


	###########Read subject data (demog and MRI)#############################

	## Read subject/scan info
	subDict = readSubjInfo(_json)

	## Create subject dataframe with all input values
	dfSub = pd.DataFrame(columns=['MRID','Age','Sex'])
	dfSub.loc[0,'MRID'] = str(subDict['MRID'])
	dfSub.loc[0,'Age'] = float(subDict['Age'])
	dfSub.loc[0,'Sex'] = str(subDict['Sex'])

	## Create dataframe to store PDF header info
	dfPat = pd.DataFrame(columns=[''])
	dfPat.loc[0] = str(subDict['MRID'])
	dfPat.loc[1] = str(subDict['Age'])
	dfPat.loc[2] = str(subDict['Sex'])
	dfPat.loc[3] = str(subDict['ExamDate'])
	dfPat.loc[4] = datetime.today().strftime('%m-%d-%Y')

	# same sex selection - all work with dfRef onwards is based on same sex
	dfRef = dfRef[dfRef.Sex == dfSub.loc[0,'Sex']]

	## Read icv, if provided, calculate icv and stored it on the subject
	icvVol = None
	dfSub['ICV'] = None
	if len(icv) == 1:
		icvVol = calcMaskVolume(icv[0])
		dfSub['ICV'] = icvVol

	# Obtain foundational objects for after this container
	roiVols, dfRef, allz, allz_num, all_MuseROIs_num, all_MuseROIs_name = calcRoiVolumes(roi[0], MUSE_ROI_Mapping, dfRef, subDict)
 
	# Add subject MUSE volumes to 
	for key in roiVols.keys():
		dfSub.loc[0, key] = roiVols[key]

	# Changes ROI column name from 702 to ICV
	dfRef['ICV'] = dfRef['702']

	## Read wmls, if provided
	wmlsVol = 0
	if len(wmls) == 1:
		wmlsVol = calcMaskVolume(wmls[0])
	dfSub['Total White Matter Hyperintensity Volume'] = wmlsVol

	### Add WMLS 604 reference datapoints to dfRef ###
	WMLSref = pd.read_csv(WMLS_Ref_Values).dropna()
	WMLSref['Date'] = pd.to_datetime(WMLSref.Date)
	WMLSref = WMLSref.sort_values(by='Date')
	# Get first-time points only
	WMLSref = WMLSref.drop_duplicates(subset=['PTID'], keep='first')
	# Binarize sex
	WMLSref['Sex'].replace(0, 'F',inplace=True)
	WMLSref['Sex'].replace(1, 'M',inplace=True)
	# same sex selection - all work with WMLSref onwards is based on same sex
	WMLSref = WMLSref[WMLSref.Sex == dfSub.loc[0,'Sex']]
 
	# Rename 604 to Total White Matter Hyperintensity Volume
	WMLSref.rename(columns={'604':'Total White Matter Hyperintensity Volume'}, inplace=True)
	Muse_ROI = copy.deepcopy(all_MuseROIs_num)

	# ICV-adjust only if ICV is available - in final version, we should throw an error and stop the pipeline
	if dfSub.loc[0,'ICV'] is not None:
		dfSub, dfRef, WMLSref, all_MuseROIs_num, all_MuseROIs_name = ICVAdjust(dfSub,dfRef,WMLSref,all_MuseROIs_num,all_MuseROIs_name)
	else:
		raise Exception('Sorry, ICV volume is not calculated. Check the ICV nii.gz file.')

	# Define which directory to save to
	out = _os.path.dirname(out_path)

	# Save values for use in other containers
	with open(_os.path.join(out,UID+'_dfSub.pkl'), 'wb') as pickle_file:
		pickle.dump(dfSub,pickle_file)
	with open(_os.path.join(out,UID+'_dfRef.pkl'), 'wb') as pickle_file:
		pickle.dump(dfRef,pickle_file)
	with open(_os.path.join(out,UID+'_WMLSref.pkl'), 'wb') as pickle_file:
		pickle.dump(WMLSref,pickle_file)
	with open(_os.path.join(out,UID+'_dfPat.pkl'), 'wb') as pickle_file:
		pickle.dump(dfPat,pickle_file)
	with open(_os.path.join(out,UID+'_allz_num.pkl'), 'wb') as pickle_file:
		pickle.dump(allz_num,pickle_file)
	with open(_os.path.join(out,UID+'_allz.pkl'), 'wb') as pickle_file:
		pickle.dump(allz,pickle_file)
	with open(_os.path.join(out,UID+'_all_MuseROIs_num.pkl'), 'wb') as pickle_file:
		pickle.dump(all_MuseROIs_num,pickle_file)
	with open(_os.path.join(out,UID+'_all_MuseROIs_name.pkl'), 'wb') as pickle_file:
		pickle.dump(all_MuseROIs_name,pickle_file)
  
	with open(_os.path.join(out,UID+'_MuseROIs_NoICVCorrection.pkl'), 'wb') as pickle_file:
		pickle.dump(Muse_ROI,pickle_file)

	return dfSub, dfRef, WMLSref, dfPat, allz_num, allz, all_MuseROIs_num, all_MuseROIs_name, Muse_ROI

    
    