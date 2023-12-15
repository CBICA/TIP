import os as _os
import numpy as np
import pandas as pd
import vtk
import SimpleITK as sitk
import sys, glob, csv
import pickle

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

from enum import Enum

#Standard atrophy buckets(chosen by Ilya)
PERCENTILE_1  = -2.326 #(2.326 sd below the mean i.e. zscore)
PERCENTILE_3  = -1.881 #(1.881 sd below the mean i.e. zscore)
PERCENTILE_6  = -1.645 #(1.645 sd below the mean i.e. zscore)
PERCENTILE_15 = -1.036 #(1.036 sd below the mean i.e. zscore)
PERCENTILE_30 = -0.5244 #(0.5244 sd below the mean i.e. zscore)

class ORIENTATION(Enum):
	RIGHT_HEMISPHERE_LATERAL = 1
	RIGHT_HEMISPHERE_MEDIAL = 2
	LEFT_HEMISPHERE_LATERAL = 3
	LEFT_HEMISPHERE_MEDIAL = 4
	BOTTOM = 5
	TOP = 6
	BASAL_GANGLIA_THALAMUS = 7
	COLORBAR = 8

def creat_title(renderer, text):
	titleProperty = vtk.vtkTextProperty()
	titleProperty.SetFontSize(20)
	titleProperty.SetBold(1)
	titleProperty.SetJustificationToCentered()

	titleMapper = vtk.vtkTextMapper()
	titleMapper.SetInput(text)
	titleMapper.SetTextProperty(titleProperty)

	titleActor = vtk.vtkActor2D()
	titleActor.SetMapper(titleMapper)
	titleActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
	titleActor.GetPositionCoordinate().SetValue(0.5, 0.85, 0.0)
	titleActor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Black"))

	# Add the titleActor to the renderer
	renderer.AddActor2D(titleActor)

def hex_to_rgb(hex: str):
	hex = hex[1:]
	assert len(hex) == 6
	rgb = [int(hex[i:i + 2], 16) for i in (0, 2, 4)]
	rgb = [x/256 for x in rgb]
	return rgb


###################### Uncomment this function to generate colorbar ##############################
# def create_scalarbar_matplotlib():
#     '''
#     create a scalarbar with matplotlib 
#     '''
#     colors = ['#bd0026', '#f03b20', '#fd8d3c', '#fecc5c', '#ffffb2']
# 	values = [-np.inf, -2.326, -1.881, -1.645, -1.036, -0.5244]

# 	# Create a ListedColormap with custom colors
# 	cmap = ListedColormap(colors)

# 	# Set up a dummy plot to create the colorbar
# 	dummy_values = np.linspace(0, 1, len(values))  # Create dummy values to plot the colorbar
# 	plt.figure(figsize=(8, 6))
# 	plt.imshow([dummy_values], cmap=cmap, aspect='auto')
# 	plt.gca().set_visible(False)  # Hide axes
# 	cbar = plt.colorbar(orientation='vertical', shrink = 0.75)

# 	# Set the custom tick positions and labels for the colorbar
# 	cbar.set_ticks(dummy_values)
# 	cbar.set_ticklabels([f'{val:.2f}' if not np.isinf(val) else '-inf' for val in values])

# 	# Set the title for the colorbar
# 	cbar.ax.set_title('Z-Score', loc='center', pad=20)

# 	# Show the plot with the colorbar
# 	plt.show()

###############################################################################################

def add_scalarbar(renderer):
 
	image_path = '/refs/colorbar.png'
	image_reader = vtk.vtkPNGReader()
	image_reader.SetFileName(image_path)
	image_reader.Update()

	image_data = image_reader.GetOutput()

	image_actor = vtk.vtkImageActor()
	image_actor.SetInputData(image_data)

	renderer.AddActor(image_actor)
 
	renderer.ResetCamera()
	camera =  renderer.GetActiveCamera()
	camera.OrthogonalizeViewUp()
	renderer.ResetCameraClippingRange()	


def get_color_TF(relabelMap,need2relabel):
	# # Define color legend #
	funcColor = vtk.vtkColorTransferFunction()
	for idx in relabelMap.keys():
		if idx in need2relabel:
			## below value chosen by Ilya as standard buckets
			if PERCENTILE_15 <= relabelMap[idx] <= PERCENTILE_30:   
				funcColor.AddRGBPoint(idx,255/255,255/255,178/255)
			elif PERCENTILE_6 <= relabelMap[idx] < PERCENTILE_15: 
				funcColor.AddRGBPoint(idx,254/255,204/255,92/255)
			elif PERCENTILE_3 <= relabelMap[idx] < PERCENTILE_6:  
				funcColor.AddRGBPoint(idx,253/255,141/255,60/255)
			elif PERCENTILE_1 <= relabelMap[idx] < PERCENTILE_3:  
				funcColor.AddRGBPoint(idx,240/255,59/255,32/255)
			elif relabelMap[idx] < PERCENTILE_1: 
				funcColor.AddRGBPoint(idx,189/255,0/255,38/255)
		else:
			funcColor.AddRGBPoint(idx,1,1,1)

	funcColor.AddRGBPoint(0,1,1,1)
	return funcColor


def get_plane_from_mri(muse_labelmap):
	nda_bg_t = sitk.GetArrayFromImage(muse_labelmap)

	bg_t = ((nda_bg_t == 59) | (nda_bg_t == 60) | (nda_bg_t == 23) | (nda_bg_t == 30) | (nda_bg_t == 36) | (nda_bg_t == 37) | (nda_bg_t == 55) | (nda_bg_t == 56) | (nda_bg_t == 57) | (nda_bg_t == 58)).astype(int)
	# print(bg_t.shape)
	cnt = np.inf
	bg_t_slice = 0

	## balanced the voxel volumn for both left and right
	for x in range(bg_t.shape[0]):
		if abs(np.sum(bg_t[:x,:,:]) - np.sum(bg_t[x:,:,:])) < cnt:
			bg_t_slice = x
			cnt = abs(np.sum(bg_t[:x,:,:]) - np.sum(bg_t[x:,:,:]))
		
	# Get transition point from left to right hemisphere from corpus callosum
	nda_all = nda_bg_t
	cc = (nda_all == 95).astype(int)

	#### test slices to get the most equitable cc split
	cnt = np.inf
	mid_slice = 0
	for y in range(cc.shape[1]):
		if abs(np.sum(cc[:,:y,:]) - np.sum(cc[:,y:,:])) < cnt:
			mid_slice = y
			cnt = abs(np.sum(cc[:,:y,:]) - np.sum(cc[:,y:,:]))
   
	print('what is bg_t_slice: ', bg_t_slice)

	return bg_t_slice, mid_slice


def get_volume(image,relabelMap,need2relabel,clip, orientation, *args):
	
	if len(args) != 0:
		bg_t_slice, mid_slice = args #get_plane_from_mri()

	#get color transfer function
	color_tf = get_color_TF(relabelMap,need2relabel)

	# Define opacity scheme 
	scalar_opacity = vtk.vtkPiecewiseFunction()
	gradient_opacity = vtk.vtkPiecewiseFunction()
	scalar_opacity.AddPoint(0, 0)
	gradient_opacity.AddPoint(0, 0)	
	for idx in relabelMap.keys():
		scalar_opacity.AddPoint(idx, 1 if idx > 0 else 0.0)
		gradient_opacity.AddPoint(idx, 1 if idx > 0 else 0.0)

	#volume property
	vol_property = vtk.vtkVolumeProperty()
	vol_property.SetColor(color_tf)
	vol_property.SetScalarOpacity(scalar_opacity)
	vol_property.SetGradientOpacity(gradient_opacity)
	vol_property.SetInterpolationTypeToNearest()
	vol_property.ShadeOn()
	vol_property.SetDiffuse(0.7)
	vol_property.SetAmbient(0.8)

	#volume mapper
	vol_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
	vol_mapper.SetInputData(image)
	if(clip):
		plane = vtk.vtkPlane()
		if(orientation == ORIENTATION.RIGHT_HEMISPHERE_MEDIAL):

			plane = vtk.vtkPlane()
			origin = list(image.GetCenter())
			origin[1] = mid_slice
			plane.SetOrigin(origin)
			plane.SetNormal(-1,0,0)

		elif(orientation == ORIENTATION.LEFT_HEMISPHERE_MEDIAL):
      
			plane = vtk.vtkPlane()
			origin = list(image.GetCenter())
			origin[1] = mid_slice
			plane.SetOrigin(origin)
			plane.SetNormal(1,0,0)
		elif(orientation == ORIENTATION.BASAL_GANGLIA_THALAMUS):

			plane = vtk.vtkPlane()
			origin = list(image.GetCenter())
			origin[2] = bg_t_slice
			plane.SetOrigin(origin)
			plane.SetNormal(0,0,-1)

		#add clipping plane to clip the volume
		vol_mapper.AddClippingPlane(plane)

	vol = vtk.vtkVolume()
	vol.SetMapper(vol_mapper)
	vol.SetProperty(vol_property)
	return vol

def save_screeenshot(rw,filename):
	windowToImageFilter = vtk.vtkWindowToImageFilter()
	windowToImageFilter.SetInput(rw)
	windowToImageFilter.Update()

	writer = vtk.vtkPNGWriter()
	writer.SetFileName(filename)
	writer.SetInputConnection(windowToImageFilter.GetOutputPort())
	writer.Write()

def setup_camera(orientation,renderer):
	if(orientation == ORIENTATION.TOP): #Top
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.75 * distance
		camera.SetPosition(focus[0],focus[1],(focus[2] + newdis))

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,-1,0)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()
	elif(orientation == ORIENTATION.BOTTOM):#Bottom
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.7 * distance
		camera.SetPosition(focus[0],focus[1],(focus[2] + newdis))

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,-1,0)
		camera.Azimuth(180)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()
	if(orientation == ORIENTATION.BASAL_GANGLIA_THALAMUS): #Basal Ganglia/Thalamus
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.75 * distance
		camera.SetPosition(focus[0],focus[1],focus[2]+newdis)

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,-1,0)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()
	if(orientation == ORIENTATION.RIGHT_HEMISPHERE_LATERAL): #Right hemisphere lateral
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.75 * distance
		camera.SetPosition((focus[0] + newdis),focus[1],focus[2])

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,0,1)
		# camera.Roll(90)
		# camera.
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()
	if(orientation == ORIENTATION.RIGHT_HEMISPHERE_MEDIAL): #Right hemisphere medial
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.7 * distance
		camera.SetPosition(focus[0],focus[1],focus[2]+newdis)
		camera.SetPosition(541.09,15.13,-4.41)
		#camera.SetPosition(focus[0],focus[1]+newdis,focus[2])

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,0,1)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()

	if(orientation == ORIENTATION.LEFT_HEMISPHERE_LATERAL): #Left hemisphere lateral
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.55 * distance
		camera.SetPosition(-(focus[0] + newdis),focus[1],focus[2])

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,0,1)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()
	if(orientation == ORIENTATION.LEFT_HEMISPHERE_MEDIAL): #Left hemisphere medial
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.6 * distance
		camera.SetPosition(-(focus[0] + newdis),focus[1],focus[2])

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,0,1)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()	


def setup_vtk_pipeline(roi,allz_num,out):
	#read muse roi as itk image once and use everywhere
	muse_itk_image = read_itk_image(roi)

	#generate dict[roi label]=zscore
	relabelMap = create_relabel_map(muse_itk_image,allz_num)

	#read muse roi image as vtk image once and use everywhere
	muse_vtk_img = get_vtk_image(roi)

	#list of roi that need relabeling for visualization
	need2relabel = get_zscore_labeled_roi_list(allz_num)

	# render_scalarbar(relabelMap,need2relabel)

	# One render window, multiple viewports.
	rw = vtk.vtkRenderWindow()
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(rw)

	rw.SetOffScreenRendering(1)
 
	# Define 8 viewports
	viewport_positions = {}
	viewport_positions[ORIENTATION.BOTTOM] = [0, 0, 0.25, 0.5]
	viewport_positions[ORIENTATION.TOP] = [0.25, 0, 0.5, 0.5]
	viewport_positions[ORIENTATION.BASAL_GANGLIA_THALAMUS] = [0.5, 0, 0.75, 0.5]
	viewport_positions[ORIENTATION.COLORBAR] = [0.75, 0, 1, 0.5]
	viewport_positions[ORIENTATION.RIGHT_HEMISPHERE_LATERAL] = [0, 0.5, 0.25, 1]
	viewport_positions[ORIENTATION.RIGHT_HEMISPHERE_MEDIAL] = [0.25, 0.5, 0.5, 1]
	viewport_positions[ORIENTATION.LEFT_HEMISPHERE_LATERAL] = [0.5, 0.5, 0.75, 1]
	viewport_positions[ORIENTATION.LEFT_HEMISPHERE_MEDIAL] = [0.75, 0.5, 1, 1]

	#create all renderers
	viewport_keys = list(viewport_positions.keys())
	list_renderers = {}
	for i in range(8):
		ren = vtk.vtkRenderer()
		rw.AddRenderer(ren)
		pos = viewport_positions[viewport_keys[i]]
		ren.SetViewport(pos[0], pos[1], pos[2], pos[3])
		ren.SetBackground(1,1,1) #white background
		ren.ResetCamera()
		list_renderers[viewport_keys[i]] = ren

	#get medial and basal ganglia slice to generate clipping plane
	bg_t_slice, mid_slice = get_plane_from_mri(muse_itk_image)

	#add actors to individual renderers
	for i in  range(len(list_renderers)):
		ren = list_renderers[viewport_keys[i]]
		if(viewport_keys[i] == ORIENTATION.RIGHT_HEMISPHERE_LATERAL):
			volume = get_volume(muse_vtk_img,relabelMap,need2relabel,False,viewport_keys[i])
			ren.AddVolume(volume)
			creat_title(ren,'Right hemisphere lateral')
			setup_camera(viewport_keys[i],ren)
		elif(viewport_keys[i] == ORIENTATION.LEFT_HEMISPHERE_LATERAL):
			volume = get_volume(muse_vtk_img,relabelMap,need2relabel,False,viewport_keys[i])
			ren.AddVolume(volume)
			creat_title(ren,'Left hemisphere lateral')
			setup_camera(viewport_keys[i],ren)
		elif(viewport_keys[i] == ORIENTATION.BOTTOM):
			volume = get_volume(muse_vtk_img,relabelMap,need2relabel,False,viewport_keys[i])
			ren.AddVolume(volume)
			creat_title(ren,'Bottom')
			setup_camera(viewport_keys[i],ren)
		elif(viewport_keys[i] == ORIENTATION.TOP):
			volume = get_volume(muse_vtk_img,relabelMap,need2relabel,False,viewport_keys[i])
			ren.AddVolume(volume)
			creat_title(ren,'Top')
			setup_camera(viewport_keys[i],ren)
		elif(viewport_keys[i] == ORIENTATION.RIGHT_HEMISPHERE_MEDIAL):
			volume = get_volume(muse_vtk_img,relabelMap,need2relabel,True,viewport_keys[i], bg_t_slice, mid_slice)
			ren.AddVolume(volume)
			creat_title(ren,'Right hemisphere medial')
			setup_camera(viewport_keys[i],ren)
		elif(viewport_keys[i] == ORIENTATION.LEFT_HEMISPHERE_MEDIAL):
			volume = get_volume(muse_vtk_img,relabelMap,need2relabel,True,viewport_keys[i], bg_t_slice, mid_slice)
			ren.AddVolume(volume)
			creat_title(ren,'Left hemisphere medial')
			setup_camera(viewport_keys[i],ren)
		elif(viewport_keys[i] == ORIENTATION.BASAL_GANGLIA_THALAMUS):
			volume = get_volume(muse_vtk_img,relabelMap,need2relabel,True,viewport_keys[i], bg_t_slice, mid_slice)
			ren.AddVolume(volume)
			creat_title(ren,'Basal Ganglia/Thalamus')
			setup_camera(viewport_keys[i],ren)
		elif(viewport_keys[i] == ORIENTATION.COLORBAR): #bottom row, last renderer for colorbar
			add_scalarbar(ren)
				
	rw.Render()
	rw.SetWindowName('Report Views')
	rw.SetSize(1280, 720)

	#save rendered image
	fname= out +'_finalvis.png'
	save_screeenshot(rw,fname)

	#iren.Start()


def read_itk_image(input_file_path):
	reader = sitk.ImageFileReader()
	reader.SetFileName ( input_file_path )
	image = reader.Execute()
	return image

def write_itk_image(img, output_file_path):
	writer = sitk.ImageFileWriter()
	writer.SetFileName ( output_file_path )
	writer.Execute ( img )

def create_relabel_map(muse_mask,roi_zscore_dict):
	stats = sitk.LabelShapeStatisticsImageFilter()
	stats.Execute(muse_mask)

	#create dict
	#dict: roi label->z score
	#dict[roi label] = z score
	merge_label_dict={}
	for i in stats.GetLabels():
		#if roi label not found in z score dict, then assign a non muse label(220)
		if str(i) not in roi_zscore_dict.keys():
			merge_label_dict[i] = 220
		else:
			merge_label_dict[i] = roi_zscore_dict[str(i)]

	return merge_label_dict

def get_vtk_image(fname):
	reader_all = vtk.vtkNIFTIImageReader()
	reader_all.SetFileName(fname)
	castFilter_all = vtk.vtkImageCast()
	castFilter_all.SetInputConnection(reader_all.GetOutputPort())
	castFilter_all.SetOutputScalarTypeToUnsignedShort()
	castFilter_all.Update()
	idbs_all = castFilter_all.GetOutput()
	return idbs_all

def get_zscore_labeled_roi_list(roi_zscore_dict):
	need2relabel = [int(k) for k,v in roi_zscore_dict.items() if float(v) <= -0.524]
	return need2relabel

def _main( roi, allz_num, pdf_path):
	UID = _os.path.basename(pdf_path.removesuffix(".pdf"))
	out = _os.path.dirname(pdf_path)
	out = out + '/' + UID
	setup_vtk_pipeline(roi,allz_num,out)

