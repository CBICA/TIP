import pickle
import os as _os
import pandas as pd
import numpy as np
from pygam import ExpectileGAM
from pygam.datasets import mcycle
from pygam import LinearGAM

import matplotlib.pyplot as plt
import matplotlib
from numpy.ma import masked_array

import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .spareAD import createSpareADplot
from .spareBA import createSpareBAplot

from .createcmap import get_continuous_cmap



# Custom AI Z-score coloring scheme
def tagAI_ID(num):
	if type(num) != str:
		if num >= 1.28:
			tag = "toolow"
		else:
			tag = "norm"
	else:
		return "norm"

	return tag

# Create color coding system based on standard deviations (-1.28 and 1.28 stds are 10th and 90th percentile, respectively)
def tagID(num):
	if type(num) != str:
		if num >= 1.28:
			tag = "toohigh"
		elif num <= -1.28:
			tag = "toolow"
		else:
			tag = "norm"
	else:
		return "norm"

	return tag

def makeTablePKL(dfRef,dfSub,path):
	names = ['Brain', 'Ventricle', 'Gray Matter', 'White Matter', 'Brainstem', 'Cerebellum', 'Hippocampus']

	tmpTable = pd.DataFrame(columns=["Brain Region", "Volume (cubic mL)", "Bilateral Z-score", "R", "R Z-score", "L", "L Z-score", "Asymmetry Index (AI)", "AI Z-score"])
	dfRef = dfRef[(dfRef['Age'] >= dfSub['Age'].values[0] -3 ) & (dfRef['Age'] <= dfSub['Age'].values[0] + 3) & (dfRef['Diagnosis_nearest_2.0'] == 'CN')]
	for key in names:
		totalnorm = round((dfSub["Total " + key + " Volume"].values[0] - dfRef["Total " + key + " Volume"].mean())/dfRef["Total " + key + " Volume"].std(),2)
		if 'Brainstem' in key:
			tmpTable.loc[len(tmpTable.index)] = [key, str(round(dfSub["Total " + key + " Volume"].values[0],1)), totalnorm, "-","-","-","-","-","-"]
		# If roi does have L/R component
		else:
			Rnorm = round((dfSub["Right " + key + " Volume"].values[0] - dfRef["Right " + key + " Volume"].mean())/dfRef["Right " + key + " Volume"].std(),2)
			Lnorm = round((dfSub["Left " + key + " Volume"].values[0] - dfRef["Left " + key + " Volume"].mean())/dfRef["Left " + key + " Volume"].std(),2)
			AInorm = abs(round((dfSub[key + " AI"].values[0] - dfRef[key + " AI"].mean())/dfRef[key + " AI"].std(),2))
			
			# Add Whole prefix to Brain ROI
			if key == 'Brain':
				tmpTable.loc[len(tmpTable.index)] = ["Whole " + key, str(round(dfSub["Total " + key + " Volume"].values[0],1)), totalnorm, str(round(dfSub["Right " + key + " Volume"].values[0],1)), Rnorm, str(round(dfSub["Left " + key + " Volume"].values[0],1)), Lnorm, str(round(dfSub[key + " AI"].values[0],2)), AInorm]
			else:
				tmpTable.loc[len(tmpTable.index)] = [key, str(round(dfSub["Total " + key + " Volume"].values[0],1)), totalnorm, str(round(dfSub["Right " + key + " Volume"].values[0],1)), Rnorm, str(round(dfSub["Left " + key + " Volume"].values[0],1)), Lnorm, str(round(dfSub[key + " AI"].values[0],2)), AInorm]

	all_entries = ""
	
	for index, rows in tmpTable.iterrows():
		tbody = ""
		tbody = '''<tr style="text-align: center;"><td><b>''' + rows["Brain Region"] + '''</b></td><td>''' + rows["Volume (cubic mL)"] + '''</td><td class=''' + tagID(rows["Bilateral Z-score"]) + '''>''' + str(rows["Bilateral Z-score"]) + '''</td><b><td>''' + rows["R"] + '''</b></td><td class=''' + tagID(rows["R Z-score"]) + '''>''' + str(rows["R Z-score"]) + '''</td><b><td>''' + rows["L"] + '''</b></td><td class=''' + tagID(rows["L Z-score"]) + '''>''' + str(rows["L Z-score"]) + '''</td><b><td>''' + rows["Asymmetry Index (AI)"] + '''</b></td><td class=''' + tagAI_ID(rows["AI Z-score"]) + '''>''' + str(rows["AI Z-score"]) + '''</td></tr>'''
		all_entries = all_entries + '''<tbody id=''' + "section" + str(index) + '''>''' + tbody + '''</tbody>'''

	string = '''<table border=1 frame=void rules=rows>
		<thead>
			<tr style="text-align: center;">
				<th scope="col">Brain Region</th>
				<th scope="col">Volume (mL<sup>3</sup>)</th>
				<th scope="col">Bilateral Z-score</th>
				<th scope="col">R</th>
				<th scope="col">R Z-score</th>
				<th scope="col">L</th>
				<th scope="col">L Z-score</th>
				<th scope="col">Asymmetry Index (AI)</th>
				<th scope="col">AI Z-score</th>
			</tr>
		</thead>
		''' + all_entries + '''
	</table>'''

	with open(path, 'wb') as f:
		pickle.dump(string, f)
  
	return string

def plotWithRef(dfRef, WMLSref, dfSub, fname, spareAD, spareBA):
	selVarlst = ['Total Brain Volume','Total Ventricle Volume','Total Gray Matter Volume',
	'Total White Matter Hyperintensity Volume','Total Brainstem Volume','Total Hippocampus Volume']
	names = ['Total Brain Volume','Total Ventricle Volume','Total Gray Matter Volume',
	'Total White Matter Hyperintensity Volume','Total Brainstem Volume','Total Hippocampus Volume']
	names.append("Spatial Pattern of Abnormality for Recognition <br>of Early Alzheimer's disease index")
	names.append("Spatial Pattern of Atrophy for Recognition <br>of Brain Aging index")

	## TODO: make specs arg. automatically adjusted to number of plots
	fig = py.subplots.make_subplots(rows=2,cols=4,subplot_titles=names,specs=[[{}, {}, {}, {}], [{}, {}, {}, {}]],vertical_spacing = 0.13,horizontal_spacing = 0.05)
	row = 1
	column = 1
	sl = False
	mark = True

	# Get only those reference subjects ±3 years from subject age
	
 
	lowlim = int(dfSub.Age.values[0]) - 3
	uplim = int(dfSub.Age.values[0]) + 3

	ADRef_ROI = dfRef.loc[dfRef['Diagnosis_nearest_2.0'] == 'AD']
	CNRef_ROI = dfRef.loc[dfRef['Diagnosis_nearest_2.0'] == 'CN']

	ADRef_WMLS = WMLSref.loc[WMLSref['Diagnosis_nearest_2.0'] == 'AD']
	CNRef_WMLS = WMLSref.loc[WMLSref['Diagnosis_nearest_2.0'] == 'CN']

	totADsubjs_ROI = ADRef_ROI.shape[0]
	totCNsubjs_ROI = CNRef_ROI.shape[0]

	totADsubjs_WMLS = ADRef_WMLS.shape[0]
	totCNsubjs_WMLS = CNRef_WMLS.shape[0]

	shownCNsubjs_ROI = CNRef_ROI.loc[(lowlim <= CNRef_ROI['Age']) & (CNRef_ROI['Age'] <= uplim)].shape[0]
	shownADsubjs_ROI = ADRef_ROI.loc[(lowlim <= ADRef_ROI['Age']) & (ADRef_ROI['Age'] <= uplim)].shape[0]

	shownCNsubjs_WMLS = CNRef_WMLS.loc[(lowlim <= CNRef_WMLS['Age']) & (CNRef_WMLS['Age'] <= uplim)].shape[0]
	shownADsubjs_WMLS = ADRef_WMLS.loc[(lowlim <= ADRef_WMLS['Age']) & (ADRef_WMLS['Age'] <= uplim)].shape[0]

	for idx, selVar in enumerate(selVarlst):
		## Only allow legend to show up for one of the plots (for display purposes)
		if mark:
			sl = True
			mark = False
		else:
			sl = False

		if selVar == 'Total White Matter Hyperintensity Volume':
			ADRef = ADRef_WMLS
			CNRef = CNRef_WMLS
		else:
			ADRef = ADRef_ROI
			CNRef = CNRef_ROI
			totADsubjs = totADsubjs_ROI
			totCNsubjs = totCNsubjs_ROI
			shownCNsubjs = shownCNsubjs_ROI
			shownADsubjs = shownADsubjs_ROI

		X_CN = CNRef.Age.values.reshape([-1,1])
		y_CN = CNRef[selVar].values.reshape([-1,1])

		X_AD = ADRef.Age.values.reshape([-1,1])
		y_AD = ADRef[selVar].values.reshape([-1,1])

		##############################################################
		########### CN values ###########
		# fit the mean model first by CV
		gam50 = ExpectileGAM(expectile=0.5).gridsearch(X_CN, y_CN)

		# and copy the smoothing to the other models
		lam = gam50.lam

		# fit a few more models
		gam90 = ExpectileGAM(expectile=0.90, lam=lam).fit(X_CN, y_CN)
		gam10 = ExpectileGAM(expectile=0.10, lam=lam).fit(X_CN, y_CN)
	
		XX_CN = gam50.generate_X_grid(term=0, n=100)
		XX90_CN = list(gam90.predict(XX_CN).flatten())
		XX50_CN = list(gam50.predict(XX_CN).flatten())
		XX10_CN = list(gam10.predict(XX_CN).flatten())
		XX_CN = list(XX_CN.flatten())

		########### AD values ###########
		# fit the mean model first by CV
		gam50 = ExpectileGAM(expectile=0.5).gridsearch(X_AD, y_AD)

		# and copy the smoothing to the other models
		lam = gam50.lam

		# fit a few more models
		gam90 = ExpectileGAM(expectile=0.90, lam=lam).fit(X_AD, y_AD)
		gam10 = ExpectileGAM(expectile=0.10, lam=lam).fit(X_AD, y_AD)
	
		XX_AD = gam50.generate_X_grid(term=0, n=100)
		XX90_AD = list(gam90.predict(XX_AD).flatten())
		XX50_AD = list(gam50.predict(XX_AD).flatten())
		XX10_AD = list(gam10.predict(XX_AD).flatten())
		XX_AD = list(XX_AD.flatten())

		CN_up5 = [abs(int(i)-(int(dfSub['Age'].values[0])+3)) for i in XX_CN]
		CN_down5 = [abs(int(i)-(int(dfSub['Age'].values[0])-3)) for i in XX_CN]
		AD_up5 = [abs(int(i)-(int(dfSub['Age'].values[0])+3)) for i in XX_AD]
		AD_down5 = [abs(int(i)-(int(dfSub['Age'].values[0])-3)) for i in XX_AD]
		setmax = max([XX90_CN[CN_up5.index(min(CN_up5))],XX90_CN[CN_down5.index(min(CN_down5))],XX90_AD[AD_up5.index(min(AD_up5))],XX90_AD[AD_down5.index(min(AD_down5))],ADRef[selVar].tolist()[0],CNRef[selVar].tolist()[0]])
		setmin = min([XX10_CN[CN_up5.index(min(CN_up5))],XX10_CN[CN_down5.index(min(CN_down5))],XX10_AD[AD_up5.index(min(AD_up5))],XX10_AD[AD_down5.index(min(AD_down5))],ADRef[selVar].tolist()[0],CNRef[selVar].tolist()[0]])
		spacer = (setmax - setmin)*.4
		setmax = setmax + spacer
		setmin = setmin - spacer

		if selVar == 'Total White Matter Hyperintensity Volume':
			#### if total white matter hyperintensity volume is avaliable, then plot the reference line, patient dot, annotation
			if dfSub[selVar].tolist()[0] != 0:
				CN_up5 = [abs(int(i)-(int(dfSub['Age'].values[0])+3)) for i in XX_CN]
				CN_down5 = [abs(int(i)-(int(dfSub['Age'].values[0])-3)) for i in XX_CN]

				setmax = max([XX90_CN[CN_up5.index(min(CN_up5))],XX90_CN[CN_down5.index(min(CN_down5))],CNRef[selVar].tolist()[0]])
				setmin = min([XX10_CN[CN_up5.index(min(CN_up5))],XX10_CN[CN_down5.index(min(CN_down5))],CNRef[selVar].tolist()[0]])
	
				## Max value to be setmax or the patient dot value 
				setmax = max(setmax, dfSub[selVar].tolist()[0])
				setmin = min(setmin, dfSub[selVar].tolist()[0])
	
				spacer = (setmax - setmin)*.4
				setmax = setmax + spacer
				setmin = max(0,setmin - spacer)
	
				fig.append_trace( go.Scatter(mode='markers', x=CNRef.Age.tolist(), y=CNRef[selVar].tolist(), legendgroup='CN', marker=dict(color='MediumBlue', size=5), opacity=0.2, name='CN Reference', showlegend=sl),row,column)
				fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX90_CN, legendgroup='CN', marker=dict(color='MediumPurple', size=10), name='90th percentile for CN', showlegend=sl),row,column)
				fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX50_CN, legendgroup='CN', marker=dict(color='MediumVioletRed', size=10), name='50th percentile for CN', showlegend=sl),row,column)
				fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX10_CN, legendgroup='CN', marker=dict(color='MediumBlue', size=10), name='10th percentile for CN', showlegend=sl),row,column)
				fig.append_trace( go.Scatter(mode='markers', x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(), legendgroup='Patient', marker=dict(color='Black', symbol = 'circle-cross-open', size=16,line=dict(color='MediumPurple', width=3)), name='Patient', showlegend=sl),row,column)
				fig.add_annotation(xref='x domain',yref='y domain',x=0.01,y=0.95, text='WMLS Volume (mL\u00b3): ' + str(round(dfSub[selVar].tolist()[0],2)), showarrow=False,row=row,col=column)

			#### if total white matter hyperintensity volume is not available, then plot the text with "WM Lesion Segmentation Failed" and then print the statement
			else:
				fig.append_trace( go.Scatter(mode='markers', x=[], y=[],  opacity=0), row=row,col=column) 
				fig.update_xaxes(visible=False, row = row, col = column)
				fig.update_yaxes(visible=False, row = row, col = column)
				fig.add_annotation(xref='x domain',yref='y domain',x=0.5,y=0.5, text='WM Lesion Segmentation Failed', font= dict(color='red'), showarrow=False,row=row,col=column)
				print('WMLS Volume is missing (check if the WMLS nii file exist)')

		else:

			CN_up5 = [abs(int(i)-(int(dfSub['Age'].values[0])+3)) for i in XX_CN] 
			CN_down5 = [abs(int(i)-(int(dfSub['Age'].values[0])-3)) for i in XX_CN]
			AD_up5 = [abs(int(i)-(int(dfSub['Age'].values[0])+3)) for i in XX_AD]
			AD_down5 = [abs(int(i)-(int(dfSub['Age'].values[0])-3)) for i in XX_AD]

			setmax = max([XX90_CN[CN_up5.index(min(CN_up5))],XX90_CN[CN_down5.index(min(CN_down5))],XX90_AD[AD_up5.index(min(AD_up5))],XX90_AD[AD_down5.index(min(AD_down5))],ADRef[selVar].tolist()[0],CNRef[selVar].tolist()[0]])
			setmin = min([XX10_CN[CN_up5.index(min(CN_up5))],XX10_CN[CN_down5.index(min(CN_down5))],XX10_AD[AD_up5.index(min(AD_up5))],XX10_AD[AD_down5.index(min(AD_down5))],ADRef[selVar].tolist()[0],CNRef[selVar].tolist()[0]])

			## Max value to be setmax or the patient dot value 
			setmax = max(setmax, dfSub[selVar].tolist()[0])
			setmin = min(setmin, dfSub[selVar].tolist()[0])

			spacer = (setmax - setmin)*.4
			setmax = setmax + spacer
			setmin = setmin - spacer

			fig.append_trace( go.Scatter(mode='markers', x=ADRef.Age.tolist(), y=ADRef[selVar].tolist(), legendgroup='AD', marker=dict(color='MediumVioletRed', size=5), opacity=0.2, name='AD Reference', showlegend=sl),row,column)
			fig.append_trace( go.Scatter(mode='markers', x=CNRef.Age.tolist(), y=CNRef[selVar].tolist(), legendgroup='CN', marker=dict(color='MediumBlue', size=5), opacity=0.2, name='CN Reference', showlegend=sl),row,column)
			fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX90_CN, legendgroup='CN', marker=dict(color='MediumPurple', size=10), name='90th percentile for CN', showlegend=sl),row,column)
			fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX50_CN, legendgroup='CN', marker=dict(color='MediumVioletRed', size=10), name='50th percentile for CN', showlegend=sl),row,column)
			fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX10_CN, legendgroup='CN', marker=dict(color='MediumBlue', size=10), name='10th percentile for CN', showlegend=sl),row,column)
			fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX90_AD, legendgroup='AD', marker=dict(color='MediumPurple', size=10), name='90th percentile for AD', line=dict(dash = 'dash'), showlegend=sl),row,column)
			fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX50_AD, legendgroup='AD', marker=dict(color='MediumVioletRed', size=10), name='50th percentile for AD', line=dict(dash = 'dash'), showlegend=sl),row,column)
			fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX10_AD, legendgroup='AD', marker=dict(color='MediumBlue', size=10), name='10th percentile for AD', line=dict(dash = 'dash'), showlegend=sl),row,column)
			fig.append_trace( go.Scatter(mode='markers', x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(), legendgroup='Patient', marker=dict(color='Black', symbol = 'circle-cross-open', size=16,line=dict(color='MediumPurple', width=3)), name='Patient', showlegend=sl),row,column)

  
		## Allow iteration through nx4 structure of plots
		if column == 4:
			row += 1
			column = 1
		else:
			column += 1

		# Update labels of subplots
		fig['layout']['xaxis{}'.format(idx+1)]['title']='<b>Age (Years)'
		fig['layout']['yaxis{}'.format(idx+1)].update(range=[setmin, setmax])
		fig['layout']['yaxis{}'.format(idx+1)]['title']='<b>ICV-normalized Volume (mL<sup>3</sup>)'
		fig['layout']['xaxis{}'.format(idx+1)]['title'].update(font=dict(size=18))
		fig['layout']['yaxis{}'.format(idx+1)]['title'].update(font=dict(size=18))

	fig = createSpareADplot(spareAD, dfSub, dfRef, fig, 2, fname)
	createSpareBAplot(spareBA, dfSub, dfRef, fig, 2, shownCNsubjs, shownADsubjs, shownCNsubjs_WMLS, totCNsubjs, totADsubjs, totCNsubjs_WMLS, fname)

# TODO: 
def makeFlagTablePKL(dfRef,mydict,MUSErois,path):
	maphemi = pd.read_csv('/refs/MUSE_ROI_Dictionary.csv')
	all_entries = ""

	# Identify L and R pairs of single ROIs
	Rinds = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'R'), 'ROI_INDEX'].values)
	Rinds_name = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'R'), 'ROI_NAME'].values)
	Linds = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'L'), 'ROI_INDEX'].values)
	Linds_name = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'L'), 'ROI_NAME'].values)

	R = dict(zip(Rinds_name, Rinds))
	L = dict(zip(Linds_name, Linds))

	AI_zscores = {}
	AI_subj = {}

	# Calculate asymmetry for each ROI-pairing - on both subject and reference z-score levels
	for i in L.keys():
		# Join into one string for right ROI
		base = i.split()
		base[0] = 'Right'
		j = ' '.join(base)

		# Calculate AI column for ROI-pairing
		AI_ref = (dfRef[str(L[i])] - dfRef[str(R[j])]).abs().div((dfRef[str(L[i])] + dfRef[str(R[j])])/2, axis = 0)
		AI = abs(MUSErois[i] - MUSErois[j])/((MUSErois[i] + MUSErois[j])/2)

		# Add to AI_zscore dictionary with format 'Base name of single ROI: AI z-score'
		base[1] = base[1].title()
		basename = ' '.join(base[1:])
		AI_zscores[basename] = abs(round((AI - AI_ref.mean())/AI_ref.std(),2))
		AI_subj[basename] = round(AI,2)

	# BELOW: 6 columns from brain volumetric into flagged volume table - rank by comparing raw AI zscore, 
	# zscore for left, z score for right to rank all ROI regions, 
	# rank by worst case among all three, exclude single ROIs without L and R

	# Identify flagrant z-scores (anything >=abs(1.28))
	priority_dict = {}
	lowercase = lambda s: s[:1].lower() + s[1:]
	for i in AI_subj.keys():
		left = 'Left ' + lowercase(i)
		right = 'Right ' + lowercase(i)

		'''if (abs(mydict[left]) >= 1.28) or (abs(mydict[right]) >= 1.28) or AI_zscores[i] >= 1.28:
			# Assign priority rank by taking max of absolutes, make dictionary, then sort into list by value
			priority = max(abs(mydict[left]),abs(mydict[right]),AI_zscores[i])

			priority_dict[i] = priority'''
		priority = max(abs(mydict[left]),abs(mydict[right]),AI_zscores[i])
		priority_dict[i] = priority

	# THEN after getting all abnormal ROI names, rank by extrema and then place into table with R vol, L vol, R-zscore, L-zscore, AI, and Z-zscore
	orderedbases = [k for k, v in sorted(priority_dict.items(), key=lambda item: item[1], reverse=True)]

	dic = {
     'Deep Nuclei': 'deep-nuclei', 
     'Limbic' : 'limbic', 
     'Parietal' : 'parietal', 
     'Frontal': 'frontal', 
     'Ventricle': 'ventricle', 
     'Deep White' : 'deep-white', 
     'Midline': 'midline',
     'Occipital': 'occipital', 
     'Cerebellum': 'cerebellum',
     'Temporal': 'temporal'}
	
	
	for index, key in enumerate(orderedbases):
		tbody = ""
		left = 'Left ' + lowercase(key)
		right = 'Right ' + lowercase(key)
		l_zscore = mydict[left]
		r_zscore = mydict[right]
		l_vol = str(round(MUSErois[left],2))
		r_vol = str(round(MUSErois[right],2))
		ai = str(AI_subj[key])
		ai_zscore = AI_zscores[key]
		lobe = list(maphemi.loc[(maphemi['ROI_NAME'] == str(left)), 'SUBGROUP_0'].values)[0]
		tbody = '''<tr style="text-align: center;"><td style="font-weight:normal;">''' + key + '''<div class="ball ''' + dic[lobe] + '''"></div></td><td style="font-weight:normal">''' + r_vol + '''</td><td class=''' + tagID(r_zscore) + ''' style="font-weight:normal">''' + str(round(r_zscore,2)) + '''</td><td style="font-weight:normal">''' + l_vol + '''</td><td class=''' + tagID(l_zscore) + ''' style="font-weight:normal">''' + str(round(l_zscore,2)) + '''</td><td style="font-weight:normal">''' + ai + '''</td><td class=''' + tagAI_ID(ai_zscore) + ''' style="font-weight:normal">''' + str(round(ai_zscore,2)) + '''</td></tr>'''
		all_entries = all_entries + '''<tbody id=''' + "section" + str(index) + '''>''' + tbody + '''</tbody>'''
  

	string = '''<table border=1 frame=void rules=rows>
		<thead>
			<tr style="text-align: center;">
				<th scope="col">Brain Region</th>
				<th scope="col">R</th>
				<th scope="col">R Z-score</th>
				<th scope="col">L</th>
				<th scope="col">L Z-score</th>
				<th scope="col">AI</th>
				<th scope="col">AI Z-score</th>
			</tr>
		</thead>
		''' + all_entries + '''
	</table>'''

	with open(path, 'wb') as f:
		pickle.dump(string, f)
  
	return string

def biomarker_main(dfSub, dfRef, WMLSref, allz_num, allz, all_MuseROIs_name, spareAD, spareBA, pdf_path):
	UID = _os.path.basename(pdf_path.removesuffix(".pdf"))
	out = _os.path.dirname(pdf_path)

	roisubsettable = makeTablePKL(dfRef,dfSub,_os.path.join(out,UID+'_roisubsettable.pkl'))
	plotWithRef(dfRef,WMLSref, dfSub, _os.path.join(out,UID+'_plot.png'), spareAD, spareBA)
	flagtable = makeFlagTablePKL(dfRef,allz,all_MuseROIs_name,_os.path.join(out,UID+'_flagtable.pkl'))
 
	return roisubsettable, flagtable

 