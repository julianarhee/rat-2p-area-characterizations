import os
import glob
import pylab as pl
import pandas as pd
import _pickle as pkl

import numpy as np
import scipy.signal
import cv2
import matplotlib.patches as patches
import seaborn as sns

from scipy import ndimage

from matplotlib.lines import Line2D
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import analyze2p.utils as hutils
#import analyze2p.plotting as pplot

import numpy as np
import numpy.fft as fft
from scipy import signal
from scipy import fftpack

import analyze2p.utils as hutils
import analyze2p.aggregate_datasets as aggr

#=================================================================== 
# Stimulus drawing functions
#===================================================================
def draw_ellipse(x0, y0, sz_x, sz_y, theta, color='b', ax=None,
                plot_centroid=False, centroid_color='b', centroid_marker='o'):
    if ax is None:
        f, ax = pl.subplots()
    if plot_centroid:
        ax.plot(x0, y0, color=centroid_color, marker=centroid_marke)
    ell = Ellipse((x0, y0), sz_x, sz_y, angle=np.rad2deg(theta))
    ell.set_alpha(0.7)
    ell.set_edgecolor(color)
    ell.set_facecolor('none')
    ell.set_linewidth(1)
    ax.add_patch(ell) 
    
    return ax


def get_image_luminances(sdf, im_arrays, pix_per_deg=16.05, resolution=[1080, 1920]):
    stim_xpos, stim_ypos = float(sdf['xpos'].unique()), float(sdf['ypos'].unique())
    lum_=[]
    for i, ((mp, sz), sg) in enumerate(sdf.groupby(['morphlevel', 'size'])):
        imname = 'M%i' % mp
        if mp==-1:
            mean_lum=float(sg['color'].values)
        else:
            imarr = im_arrays[imname]
            iscr  = draw_stimulus_to_screen(imarr, size_deg=sz, 
                                        stim_pos=(stim_xpos, stim_ypos),
                                        pix_per_deg=pix_per_deg, 
                                        resolution=resolution)
            mean_lum = iscr.mean()/255.
        lum_.append(pd.DataFrame({'config': sg.index, 'name': imname, 
                                    'size': sz, 'morphlevel': mp, 'lum': mean_lum}, index=[i]))
    lumdf=pd.concat(lum_)
    
    return lumdf

def stimsize_poly(sz, xpos=0, ypos=0):
    from shapely.geometry import box

    ry_min = ypos - sz/2.
    rx_min = xpos - sz/2.
    ry_max = ypos + sz/2.
    rx_max = xpos + sz/2.
    s_blobs = box(rx_min, ry_min, rx_max, ry_max)
    
    return s_blobs

#def stim_to_screen(stim, stim_xpos, stim_ypos,  size_deg, 
#                   pix_per_deg=16.05, resolution=[1920, 1080]):
#    stim_screen, stim_extent = draw_stimulus_to_screen(stim, 
#                                        size_deg=size_deg, 
#                                        stim_pos=(stim_xpos, stim_ypos),
#                            pix_per_deg=pix_per_deg, resolution=resolution[::-1])
#    return stim_screen

def draw_stimulus_to_screen(stimulus_im, size_deg=30., stim_pos=(0, 0),
                            pix_per_deg=16.05, resolution=[1080, 1920]):
    
    # Reshape stimulus to what it would be at size_deg
    im_r = resize_image_to_coords(stimulus_im, size_deg=size_deg)

    # Get extent of resized image, relative to stimulus coordinates
    stim_xpos, stim_ypos = stim_pos
    stim_extent=[-im_r.shape[1]/2. + stim_xpos, im_r.shape[1]/2. + stim_xpos, 
            -im_r.shape[0]/2. + stim_ypos, im_r.shape[0]/2. + stim_ypos]

    # Create array (dims=resolution)
    stim_screen = place_stimulus_on_screen(stimulus_im, stim_extent, 
                                             resolution=resolution)
    
    return stim_screen #, stim_extent

def resize_image_to_coords(im, size_deg=30, pix_per_deg=16.05, aspect_scale=1.747):
    '''
    Take original image (in pixels) and scale it to specified size for screen.
    Return resized image in pixel space.
    '''
    #print(pix_per_deg)
    ref_dim = max(im.shape)
    resize_factor = ((size_deg*pix_per_deg) / ref_dim ) / pix_per_deg
    scale_factor = resize_factor * aspect_scale
    
    imr = cv2.resize(im, None, fx=scale_factor, fy=scale_factor)
    
    return imr


def place_stimulus_on_screen(im, stim_extent, resolution=[1080, 1920]):
    '''
    Place re-sized image (resize_image_to_coors()) onto the screen at specified res.
    extent: (xmin, xmax, ymin, ymax)
    ''' 
    lin_x, lin_y = hutils.get_lin_coords(resolution=resolution)
    
#    xx, yy = np.where(abs(lin_x-extent[0])==abs(lin_x-extent[0]).min())
#    xmin=int(np.unique(yy))
#
#    xx, yy = np.where(abs(lin_x-extent[1])==abs(lin_x-extent[1]).min())
#    xmax=int(np.unique(yy))
#
#    xx, yy = np.where(abs(lin_y-extent[2])==abs(lin_y-extent[2]).min())
#    ymin = resolution[0] - int(np.unique(xx))
#
#    xx, yy = np.where(abs(lin_y-extent[3])==abs(lin_y-extent[3]).min())
#    ymax = resolution[0] - int(np.unique(xx))

#    nw = xmax - xmin
#    nh = ymax - ymin
#    im_r2 = cv2.resize(im, (nw, nh))
#

    az_values = lin_x[0,:]
    linx_min, linx_max = min(az_values), max(az_values)
    xdim_pix = hutils.convert_range([stim_extent[0], stim_extent[1]], 
                         oldmin=linx_min, oldmax=linx_max, 
                         newmin=0, newmax=resolution[1]) #liny_min)
    el_values = lin_y[:, 0][::-1] # flip so starts with neg in array
    liny_min, liny_max = min(el_values), max(el_values)
    ydim_pix = hutils.convert_range([stim_extent[2], stim_extent[3]], 
                         oldmin=liny_min, oldmax=liny_max, 
                         newmin=0, newmax=resolution[0]) #liny_min)
    xmin, xmax = (int(round(i)) for i in xdim_pix)
    ymin, ymax = (int(round(i)) for i in ydim_pix)
    nw = int(xmax-xmin)
    nh = int(ymax-ymin)
    im_r2 = cv2.resize(im, (nw, nh))

    sim_screen = np.zeros(lin_x.shape)
    # Check x-dimension
    im0 = im_r2.copy()
    if xmax>sim_screen.shape[1]: # too far on right side
        trim = xmax - sim_screen.shape[1] # how much beyond are we
        im0 = im_r2[:, 0:(nw-trim)] # trim image 
        xmax = sim_screen.shape[1]
    if xmin<0: # Too far on left side
        trim = abs(xmin)
        im0 = im_r2[:, trim:]
        xmin = 0
    # Check vertical dimension
    im1 = im0.copy()
    if ymax>sim_screen.shape[0]: # too far on top
        trim = ymax - sim_screen.shape[0]
        im1 = im0[trim:, :]
        ymax = sim_screen.shape[0]
    if ymin<0: # too far on bottom
        trim = abs(ymin)
        im1 = im0[0:(nh-trim), :]
        ymin = 0

    sim_screen[ymin:ymax, xmin:xmax] = np.flipud(im1)

    return sim_screen

def convert_fitparams_to_pixels(rid, curr_rfs, pix_per_deg=16.06,
                                resolution=[1080, 1920],
                                convert_params=['x0', 'y0', 'fwhm_x', 'fwhm_y', 'std_x', 'std_y']):
    '''
    RF fit params in degrees, convert to pixel space for drawing
    '''
    lin_x, lin_y = hutils.get_lin_coords(resolution=res)

    # Get position
    ctx = curr_rfs['x0'][rid]
    cty = curr_rfs['y0'][rid]

    # Convert to deg
    _, yy = np.where(abs(lin_x-ctx)==abs(lin_x-ctx).min())
    x0=int(np.unique(yy))
    xx, _ = np.where(abs(lin_y-cty)==abs(lin_y-cty).min())
    y0=res[0]-int(np.unique(xx))

    # Get sigmax-y:
    sigx = curr_rfs['fwhm_x'][rid]
    sigy = curr_rfs['fwhm_y'][rid]

    sz_x = sigx*pix_per_deg #*.5
    sz_y = sigy*pix_per_deg #*.5
    
    theta = curr_rfs['theta'][rid]
    
    return x0, y0, sz_x, sz_y, theta

def load_stimuli(rename_morphs=True, return_paths=False,
                 src_dir='/n/coxfs01/julianarhee/stimuli/images'):

    #stimulus_dir = os.path.join(root, stimulus_path)
    # Get image paths:

    object_list = ['N1', 'M14', 'M27', 'M40', 'M53', 'M66', 'M79', 'M92', 'N2']
   
    image_paths = [os.path.join(src_dir, 'Blob_%s_CamRot_y0.png' % obj) \
                    if obj in ['N1', 'N2'] \
                    else os.path.join(src_dir, \
                    'morph%i_CamRot_y0.png' % int(obj[1:])) \
                    for obj in object_list]
    found_paths = [i for i in image_paths if os.path.exists(i)]
    assert len(found_paths)>0, "Missing stimuli in:\n  %s" % src_dir
    images = {}
    for object_name, impath in zip(object_list, image_paths):
        im = cv2.imread(impath)
        if rename_morphs and object_name in ['N1', 'N2']:
            oname = 'M0' if object_name=='N1' else 'M106'
        else:
            oname = object_name
        images[oname] = im[:, :, 0]
    if return_paths:
        return images, image_paths
    else: 
        return images


def load_stimuli_home(root='/n/home00/juliana.rhee', rename_morphs=True,
                return_paths=False,
                 stimulus_path='Repositories/protocols/physiology/stimuli/images'):

    stimulus_dir = os.path.join(root, stimulus_path)
    # stimulus_dir = '/n/home00/juliana.rhee/Repositories/protocols/physiology/stimuli/images'

    # Get image paths:
    #object_list = ['D1', 'D2']
    object_list = ['D1', 'M14', 'M27', 'M40', 'M53', 'M66', 'M79', 'M92', 'D2']
    
    image_paths = []
    for obj in object_list:
        stimulus_type = 'Blob_%s_Rot_y_fine' % obj
        image_paths.extend(glob.glob(os.path.join(stimulus_dir, stimulus_type, '*_y0.png')))
    print("%i images found for %i objects" % (len(image_paths), len(object_list)))
    assert len(image_paths)>0, "No stimuli in:\n  %s" % stimulus_dir
    images = {}
    for object_name, impath in zip(object_list, image_paths):
        im = cv2.imread(impath)
        if rename_morphs and object_name in ['D1', 'D2']:
            oname = 'M0' if object_name=='D1' else 'M106'
        else:
            oname = object_name
        images[oname] = im[:, :, 0]
    #print("im shape:", images['M14'].shape)
    if return_paths:
        return images, image_paths
    else: 
        return images

def gabor_patch(size, sf=None, lambda_=None, theta=90, sigma=None, 
                phase=0, trim=.005, pix_per_degree=16.05,return_grating=False):
    """Create a Gabor Patch

    size : int
        Image size (n x n)

    lambda_ : int
        Spatial frequency (px per cycle)

    theta : int or float
        Grating orientation in degrees

    sigma : int or float
        gaussian standard deviation (in pixels)

    phase : float
        0 to 1 inclusive
    """
    assert not (sf is None and lambda_ is None), "Must specify sf or lambda)_"
    
    deg_per_pixel=1./pix_per_degree
    lambda_ = 1./(sf*deg_per_pixel) # cyc per pixel (want: pix/cyc)
    
    if sigma is None:
        sigma = max(size)
    
    sz_y, sz_x = size
    
    # make linear ramp
    X0 = (np.linspace(1, sz_x, sz_x) / sz_x) - .5
    Y0 = (np.linspace(1, sz_y, sz_y) / sz_y) - .5

    # Set wavelength and phase
    freq = sz_x / float(lambda_)
    phaseRad = phase * 2 * np.pi

    # Make 2D grating
    Ym, Xm = np.meshgrid(X0, Y0)

    # Change orientation by adding Xm and Ym together in different proportions
    thetaRad = (theta / 360.) * 2 * np.pi
    Xt = Xm * np.cos(thetaRad)
    Yt = Ym * np.sin(thetaRad)
    grating = np.sin(((Xt + Yt) * freq * 2 * np.pi) + phaseRad)

    # 2D Gaussian distribution
    gauss = np.exp(-((Xm ** 2) + (Ym ** 2)) / (2 * (sigma / float(sz_x)) ** 2))

    # Trim
    gauss[gauss < trim] = 0

   
    if return_grating:
        print("returning grating")
        return grating
    else: 
        return grating * gauss



#=================================================================== 
# Image processing
# ===================================================================

def get_bbox_around_nans(rpatch, replace_nans=True, return_indices=False):
    bb_xmax, bb_ymax = np.max(np.where(~np.isnan(rpatch)), 1)
    bb_xmin, bb_ymin = np.min(np.where(~np.isnan(rpatch)), 1)
    # print(bb_xmax, bb_ymax)

    tp = rpatch[bb_xmin:bb_xmax, bb_ymin:bb_ymax]
    bb_patch=tp.copy()
    if replace_nans:
        bb_patch[np.where(np.isnan(tp))] = -1
        
    if return_indices:
        return bb_patch, (bb_xmin, bb_xmax, bb_ymin, bb_ymax)
    else:
        return bb_patch

def blur_mask(mask, ks=None):
    mask_p = mask.astype(float)
    if ks is None:
        ks = int(min(mask_p.shape)/2.)+1
    mask_win = cv2.GaussianBlur(mask_p, (ks, ks), 0)
    return mask_win

def orig_patch_blurred(im_screen, x0, y0, sz_x, sz_y, rf_theta, ks=101):
    '''
    rf_theta is in degrees
    '''
    curr_rf_theta = np.deg2rad(rf_theta)
    curr_rf_mask, curr_rf_bbox = rf_mask_to_screen(x0, y0, sz_x, sz_y, curr_rf_theta,
                                              resolution=im_screen.shape)
    msk_xmin, msk_xmax, msk_ymin, msk_ymax = curr_rf_bbox
    rf_mask_patch = curr_rf_mask[msk_xmin:msk_xmax, msk_ymin:msk_ymax]

    # Bitwise AND operation to black out regions outside the mask
    result = im_screen * curr_rf_mask
    rf_patch = result[msk_xmin:msk_xmax, msk_ymin:msk_ymax]

    # blur RF edges
    blurred_mask = blur_mask(curr_rf_mask, ks=ks)
    win_patch = blurred_mask*im_screen

    # crop bbox
    win_patch = win_patch[msk_xmin:msk_xmax, msk_ymin:msk_ymax]

    return result, rf_patch, win_patch


#=================================================================== 
# RF masking
# ===================================================================
def rfdframs_df(curr_rfs):
    '''Parals for creating RF ellipse to screen. Index = cell rids'''
    params = ['cell', 'x0', 'y0', 'fwhm_x', 'fwhm_y', 'theta']
    rfs_ = curr_rfs[params]
    rfs_.index = rfs_['cell'].values
    return rfs_

def get_lin_match(v, lin_x, axis=1):
    return int(np.unique(np.where(abs(lin_x-v)==abs(lin_x-v).min())[axis]))

def params_deg_to_pixels(rfs_, pix_per_deg=16.05, resolution=[1920, 1080]):

    lin_x, lin_y = hutils.get_lin_coords(resolution=resolution[::-1])

    rfs_.index = rfs_['cell'].values
    rfs_.loc[rfs_.index, 'x0_pix'] = [get_lin_match(v, lin_x, axis=1) \
                      for v in rfs_['x0'].values]
    rfs_.loc[rfs_.index, 'y0_pix'] = [resolution[1]-get_lin_match(v, lin_y, axis=0) \
                      for v in rfs_['y0'].values]

    rfs_.loc[rfs_.index, 'fwhm_x_pix'] =  rfs_['fwhm_x'] * pix_per_deg
    rfs_.loc[rfs_.index, 'fwhm_y_pix'] =  rfs_['fwhm_y'] * pix_per_deg
    return rfs_


from shapely.geometry import Polygon, MultiPolygon
def image_to_poly(im):
    contours, hierarchy = cv2.findContours(im, cv2.RETR_CCOMP, 
                                       cv2.CHAIN_APPROX_SIMPLE)
    contours = map(np.squeeze, contours)  # removing redundant dimensions
    polygons = map(Polygon, contours) 
    poly = MultiPolygon(polygons)
    return poly

# --------------------------------------------------------------------
# Receptive field shapes
# --------------------------------------------------------------------
def load_rfpolys(fit_desc, combine_method='average',
        aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    rf_polys=None
    check_rfs={}

    dst_dir = os.path.join(aggregate_dir, 'receptive-fields', 'dataframes')
    poly_fpath = os.path.join(dst_dir, 'polys_%s_%s.pkl' % (fit_desc, combine_method))
    try:
        with open(poly_fpath, 'rb') as f:
            res = pkl.load(f)
        check_rfs = res['check_rfs']
        rf_polys0 = res['POLYS'].reset_index(drop=True)
        check_dup = rf_polys0[['datakey', 'cell']].drop_duplicates().index.tolist()
        rf_polys= rf_polys0.loc[check_dup]

    except Exception as e:
        raise(e)

    return rf_polys, check_rfs


def update_rfpolys(rfdf, fit_desc, combine_method='average', create_new=False,
                  aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    # Set output file
    dst_dir = os.path.join(aggregate_dir, 'receptive-fields', 'dataframes')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    poly_fpath = os.path.join(dst_dir, 'polys_%s_%s.pkl' % (fit_desc, combine_method))
    # poly_fpath = os.path.join(dst_dir, 'average_polys_%s.pkl' % fit_desc)

    POLYS=None; check_rfs={};
    if not create_new:
        # Load existing   
        POLYS, check_rfs = load_rfpolys(fit_desc, combine_method=combine_method,
                                aggregate_dir=aggregate_dir)

    # Process all cells in fov
    cols = [c for c in rfdf.columns if c!='visual_area']
    by_dkey = rfdf[cols].drop_duplicates()
    # Add new, if needed
    check_these={}
    poly_list=[]
    POLYS=None
    for dk, curr_rfs in by_dkey.groupby('datakey'):
        # Get the cells we need 
        curr_polys, curr_checks = get_rf_polys(curr_rfs, check_invalid=True)
        if len(curr_checks)>0:
            check_these[dk]= curr_checks
        curr_polys['datakey'] = dk
        poly_list.append(curr_polys)
        print("    adding %s, %i" % (dk, len(curr_polys)))
    
    if len(poly_list)>0:
        POLYS = pd.concat(poly_list, axis=0)

    check_rfs.update(check_these)
    # Save
    res = {'check_rfs': check_rfs, 'POLYS': POLYS}
    with open(poly_fpath, 'wb') as f:
        pkl.dump(res, f, protocol=2)
            
    return POLYS, check_rfs



def rf_to_screen(rid, rfs_, resolution=[1920, 1080]):
    ''' just returns mask'''
    params = ['x0', 'y0', 'fwhm_x', 'fwhm_y', 'theta']
    if 'x0_pix' not in rfs_.columns:
        rfs_ = params_deg_to_pixels(rfs_.copy(), resolution=resolution)
    params_pix = ['%s_pix' % p if p!='theta' else p for p in params]
    x0, y0, fwhm_x, fwhm_y, theta = rfs_.loc[rid, params_pix]

    # Create mask
    curr_rf_mask = np.zeros(resolution[::-1]).astype(np.uint8) 
    curr_rf_mask=cv2.ellipse(curr_rf_mask, (int(x0), int(y0)), 
                     (int(fwhm_x/2), int(fwhm_y/2)), 
                     np.rad2deg(theta), 
                     startAngle=360, endAngle=0, color=1, thickness=-1)
    return curr_rf_mask


def rf_mask_to_screen(x0, y0, fwhm_x, fwhm_y, theta, resolution=[1080, 1920]):
    '''
    Return mask on screen (pixels), and bbox around mask. 
    x0, y0 in pixel coords.
    
    Uses cv2.ellipse(), which expects std (not fwhm), so divide fwhm by 2.
 
    theta is in RAD.
    
    '''
    # Create mask
    curr_rf_mask = np.zeros(resolution).astype(np.uint8) 
    curr_rf_mask=cv2.ellipse(curr_rf_mask, (int(x0), int(y0)), 
                     (int(fwhm_x/2), int(fwhm_y/2)), 
                     np.rad2deg(theta), 
                     startAngle=360, endAngle=0, color=1, thickness=-1)

    mask_nan = curr_rf_mask.copy().astype(float)
    mask_nan[curr_rf_mask==0]=np.nan
    mask_bb, curr_rf_bbox = get_bbox_around_nans(mask_nan, 
                                    replace_nans=False, return_indices=True)
    return curr_rf_mask, curr_rf_bbox


def get_stimulus_polys(dk, experiment='blobs', create_new=False,
                    return_onscreen=False, verbose=False,
                    rootdir='/n/coxfs01/2p-data'):
    ''' 
    Loads or creates shapely Polygons for all images on screen.
    Args
    
    return_onscreen: (bool)
        Set to also return dict with images convert to screen (pixels)
    
    Returns
    
    stim_polys: (pd.DataFrame)
        Each row is a poly, and includes config as str of:
        morph-size-xpos-ypos
    '''


    # Set output dir and file
    session, animalid, fovnum = hutils.split_datakey_str(dk)
    experiment_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                        'FOV%i_*' % fovnum, 'combined_%s_static' % experiment))[0] 
    poly_fpath = os.path.join(experiment_dir, 'stimuli_polys.pkl')
    try:
        with open(poly_fpath, 'rb') as f:
            stim_polys = pkl.load(f)
    except Exception as e:
        create_new=True

    if create_new or return_onscreen: 
        print("... (%s) creating stimulus polys" % dk)
        # Get image stimuli
        images = load_stimuli()

        # Stimulus info
        #stim_xpos, stim_ypos = aggr.get_stimulus_coordinates(dk, experiment)
        sdf = aggr.get_stimuli(dk, experiment, match_names=True)
        stimdf = sdf[sdf['morphlevel']!=-1].copy()

        # stim_sizes = sorted(sdf['size'].unique())
        # Convert to screen (pixels)
        pix_per_deg, screen_res = hutils.get_pixels_per_deg()
        stim_screen={}
        p_list=[]
        for (mp, sz, xp, yp) in \
                    stimdf[['morphlevel', 'size', 'xpos', 'ypos']].values:
            im = images['M%i' % mp].copy()
            onscreen  = draw_stimulus_to_screen(im, size_deg=sz, 
                                        stim_pos=(xp, yp),
                                        pix_per_deg=pix_per_deg, 
                                        resolution=screen_res[::-1]) #[1080, 1920]
            blob_poly = image_to_poly(onscreen.astype(np.uint8))

            stim_screen[(mp, sz)] = onscreen
            p_list.append(pd.DataFrame({'poly': blob_poly, 
                                        'stimulus': '%i_%i_%i_%i' %(mp, sz, xp, yp),
                                        'morphlevel': mp, 'size': sz,
                                        'xpos': xp, 'ypos': yp}))
        stim_polys = pd.concat(p_list, axis=0).reset_index(drop=True)
        # Save
        with open(poly_fpath, 'wb') as f:
            pkl.dump(stim_polys, f, protocol=2)
        if verbose:
            print("    saved: %s" % poly_fpath) 
    if return_onscreen:
        return stim_polys, stim_screen
    else: 
        return stim_polys



def get_rf_polys(curr_rfs, check_invalid=False, resolution=[1920, 1080]):
    
    ''' get dataframe of all rfs into polys created from fit params'''
    p_list=[]
    check_rfs=[]
    rfs_ = params_deg_to_pixels(curr_rfs)

    roi_list = rfs_['cell'].unique()
    for ri in roi_list:
        rf_screen = rf_to_screen(ri, rfs_, resolution=resolution)
        rpoly = image_to_poly(rf_screen.astype(np.uint8))
        if not rpoly.is_valid:
            rpoly = rpoly.buffer(0)
            check_rfs.append((ri, rpoly))

        p_list.append(pd.DataFrame({'poly': rpoly, 'cell': ri}, index=[ri]))

    rfpolys = pd.concat(p_list, axis=0).reset_index(drop=True)    

    if check_invalid:
        return rfpolys, check_rfs
    else:
        return rfpolys

def calculate_rf_overlap(rfpoly, stimpoly, roi_id='roi', stim_id='stim'):
    #roi_id, rfpoly = rfpoly_tuple
    #stim_id, stimpoly = stimpoly_tuple

    area_of_smaller = min([rfpoly.area, stimpoly.area])
    overlap_area = rfpoly.intersection(stimpoly).area
    rf_overlap = overlap_area/rfpoly.area #area_of_smaller
    rel_overlap = overlap_area/area_of_smaller

    odf = pd.DataFrame({'cell': roi_id,
                        'stimulus': stim_id,
                        'area_overlap': overlap_area,
                        'rf_overlap': rf_overlap,
                        'relative_overlap': rel_overlap}, index=[0])
    
    return odf



def cell_overlap_with_stimuli(rid, rf_poly, stim_polys):
    import analyze2p.receptive_fields.utils as rfutils
    '''For 1 cell, calculate its overlap with all stimuli
    rf_poly: shapely Polygon
    stim_polys: pd.DataFrame of all polys
    
    returns overlaps as pd.DataFrame (use perc_overlap)
    '''
    o_list=[]
    for (mp, sz, xp, yp), cval in stim_polys.groupby(['morphlevel', 'size', 'xpos', 'ypos']):
        stim_key = '%i_%i_%i_%i' % (mp, sz, xp, yp)
        o = calculate_rf_overlap(rf_poly, cval['poly'].iloc[0],
                                 roi_id=rid, stim_id=stim_key)
        o_list.append(o)
    df_ = pd.concat(o_list, axis=0).reset_index(drop=True)
    df_ = df_.rename(columns={'poly1': 'cell', 'poly2': 'stimulus'})
    
    return df_

def calculate_overlaps_fov(dk, curr_rfs, check_invalid=False, 
                    experiment='blobs', resolution=[1920, 1080]):

    stim_polys = get_stimulus_polys(dk, experiment)
    rf_polys=None; check_rfs=None;
    if 'poly' not in curr_rfs.columns or ('poly' in curr_rfs.columns and None in curr_rfs['poly'].values): #is None:
        print('    getting rf polys')
        rf_polys, check_rfs = get_rf_polys(curr_rfs, check_invalid=True, 
                            resolution=resolution)
    else:
        rf_polys = curr_rfs[['cell', 'poly']].copy()
    overlaps = pd.concat([cell_overlap_with_stimuli(ri, rf_poly, stim_polys)\
                for ri, rf_poly in rf_polys[['cell', 'poly']].values])

    if check_invalid:
        return rf_polys, check_rfs
    else:
        return overlaps


