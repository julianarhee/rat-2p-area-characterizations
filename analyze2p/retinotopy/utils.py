import glob
import os
import cv2
import glob
import re
import json
import h5py
import traceback

import matplotlib as mpl
import tifffile as tf
import pandas as pd
import pylab as pl
import seaborn as sns
import dill as pkl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import misc,interpolate,stats,signal
import analyze2p.utils as hutils
#from py3utils import natural_keys

import analyze2p.extraction.rois as roiutils
import analyze2p.utils as hutils
import analyze2p.plotting as pplot

import scipy.stats as spstats

# Data selection
def get_average_mag_across_pixels(datakey, retinorun=None,
                                rootdir='/n/coxfs01/2p-data'):
    '''
    Averages magnitude across trials and cond, and
    get average mag value for a given retino run. 
    '''
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    fov='FOV%i_zoom2p0x' % fovn

    magratios=[]
    search_run = 'retino_run' if retinorun is None else retinorun    
    retinoruns = [os.path.split(r)[-1] for r \
                    in glob.glob(os.path.join(rootdir, animalid, session, 
                    'FOV%i_*' % fovn, '%s*' % search_run))] 
    for retinorun in retinoruns:
        try:
            retinoid, RETID = load_retino_analysis_info(\
                                datakey, retinorun, use_pixels=True)
            assert RETID is not None, \
                "Error loading analysis: %s (%s)" % (retinorun, datakey)

            magratio, phase, trials_by_cond = fft_results_by_trial(RETID)
            mean_mag = magratio.mean(axis=0).mean()
            magratios.append((retinorun, mean_mag))
        except Exception as e:
            print(e)
            continue

    return pd.DataFrame(magratios)

def select_strongest_retinorun(projection_df):
    d_=[]
    #m_=[]
    for (varea, dkey), g in projection_df.groupby(['visual_area', 'datakey']):
        if len(g['retinorun'].unique())>1:
            session, animalid, fovn = dkey.split('_')
            fov = 'FOV%i_zoom2p0x' % int(fovn[3:])
            magratios = get_average_mag_across_pixels(animalid, session, fov)
    #         means0 = pd.DataFrame({'retinorun': [m[0] for m in magratios],
    #                                'magratio': [m[1] for m in magratios]})
    #         means2 = g.groupby(['retinorun']).mean().reset_index()[['retinorun', 'R2']]
    #         means = means0.merge(means2)
    #         means = putils.add_meta_to_df(means, {'visual_area': visual_area,'datakey': datakey})
    #         m_.append(means)
            if magratios[0][1] > magratios[1][1]:
                d_.append(g[g['retinorun']==magratios[0][0]])
            else:
                d_.append(g[g['retinorun']==magratios[1][0]])
        else:
            d_.append(g)
    df = pd.concat(d_, axis=0).reset_index(drop=True).drop_duplicates()
    
    return df




# -----------------------------------------------------------------------------
# Map funcs
# -----------------------------------------------------------------------------
# MAPS: Specific to 2p
def get_final_maps(magratios_soma, phases_soma, trials_by_cond=None, 
                    mag_thr=0.01, delay_thr=0.5, verbose=False,
                   dims=(512, 512), ds_factor=2, use_pixels=False):
    if mag_thr is None:
        mag_thr = -np.inf
    # Get absolute maps from conditions
    magmaps, absolute_az, absolute_el, delay_az, delay_el = absolute_maps_from_conds(
                            magratios_soma, phases_soma, trials_by_cond=trials_by_cond,
                            mag_thr=mag_thr, dims=dims, #(d1_orig, d2_orig),
                            ds_factor=ds_factor, return_map=use_pixels, verbose=verbose)
    
    # #### Filter where delay map is not uniform (Az v El)
    filt_az, filt_el = filter_by_delay_map(absolute_az, absolute_el,
                                delay_az, delay_el,
                                delay_map_thr=delay_thr, return_delay=False)
    phases_by_cond = pd.DataFrame({'phase_az': filt_az, 'phase_el': filt_el})

    #### Mean mag ratios
    if isinstance(magmaps, dict):
        magmaps = pd.concat(magmaps, axis=1)
    mags_by_cond = pd.DataFrame(magmaps[['right', 'left']]\
                        .mean(axis=1), columns=['mag_az'])
    mags_by_cond['mag_el'] = magmaps[['top', 'bottom']].mean(axis=1)
    
    #### Reprt
    ntotal = absolute_az.shape[0]
    n_pass_magthr = absolute_az.dropna().shape[0]
    n_pass_delaythr = filt_az.dropna().shape[0]
    if verbose:
        print("Total: %i\n After mag_thr (%.3f): %i\n After delay_thr (%.3f): %i" \
          % (ntotal, mag_thr, n_pass_magthr, delay_thr, n_pass_delaythr))
    
    df = pd.concat([phases_by_cond, mags_by_cond], axis=1)
    
    return df #filt_az, filt_e, 

def correct_phase_wrap(phase):

    corrected_phase = phase.copy()

    corrected_phase[phase<0] =- phase[phase<0]
    corrected_phase[phase>0] = (2*np.pi) - phase[phase>0]

    return corrected_phase

def arrays_to_maps(magratio, phase, trials_by_cond, use_cont=False,
                            dims=(512, 512), ds_factor=2, cond='right',
                            mag_thr=None, mag_perc=0.05, return_map=True):
    if mag_thr is None:
        mag_thr = -np.inf # Set to neg infinity for NO filter
        #mag_thr = magratio.max().max()*mag_perc

    currmags = magratio[trials_by_cond[cond]].copy()
    currmags_mean = currmags.mean(axis=1)
    currmags_mean.loc[currmags_mean<mag_thr] = np.nan 
    #currmags_mean = means_[means_>=mag_thr]

    if return_map:
        d1 = int(dims[0] / ds_factor)
        d2 = int(dims[1] / ds_factor)
        # print(d1, d2)
        currmags_map = np.reshape(currmags_mean.values, (d1, d2))
    else:
        currmags_map = currmags_mean.copy()
        
    currphase = phase[trials_by_cond[cond]].copy() #.loc[currmags_mean.index]
    currphase.loc[currmags_mean[np.isnan(currmags_mean)].index, trials_by_cond[cond]] = np.nan
    #currphase_mean = stats.circmean(currphase, low=-np.pi, high=np.pi, axis=1, nan_policy='omit')
    #currphase_mean_c = correct_phase_wrap(currphase_mean)
    non_nan_ix = currphase.dropna().index #.tolist()
    print("%i non-nan of %i (thr=%.3f)" % (len(non_nan_ix), len(currphase), mag_thr))
    currphase_mean0 = stats.circmean(currphase.dropna(), low=-np.pi, high=np.pi, axis=1) #, nan_policy='omit')
    currphase_mean_c0 = correct_phase_wrap(currphase_mean0)
    currphase_mean_c = pd.DataFrame(data=np.ones(len(currphase),)*np.nan, index=currphase.index)
    currphase_mean_c.loc[non_nan_ix, 0] = currphase_mean_c0

    if return_map:
        #currphase_mean_c[np.isnan(currmags_mean)] = np.nan
        currphase_map_c = np.reshape(currphase_mean_c.values, (d1, d2))
    else:
        currphase_map_c = currphase_mean_c.copy()
        
    return currmags_map, currphase_map_c, mag_thr

def absolute_maps_from_conds(magratio, phase, trials_by_cond=None, mag_thr=0.01,
                        dims=(512, 512), ds_factor=2, return_map=True, verbose=False):
    '''Calculate absolute maps and delay maps from conditions'''
    if mag_thr is None:
        mag_thr = -np.inf

    use_cont=False # doens't matter, should be equiv now
    magmaps = {}
    phasemaps = {}
    magthrs = {}
    if trials_by_cond is not None:
        for cond in trials_by_cond.keys():
            magmaps[cond], phasemaps[cond], magthrs[cond] = arrays_to_maps(
                                                magratio, phase, trials_by_cond,
                                                cond=cond, use_cont=use_cont,
                                                mag_thr=mag_thr, dims=dims,
                                                ds_factor=ds_factor, 
                                                return_map=return_map)
    else:
        non_nan_ix = magratio[magratio<mag_thr].dropna().index.tolist()
        magmaps = magratio.copy()
        magmaps.loc[non_nan_ix] = np.nan
        # Make continuous, from 0 to 2pi 
        phasemaps = phase.apply(correct_phase_wrap) 
        # Don't include low-responding pix for phase calc
        phasemaps.loc[non_nan_ix] = np.nan
        
    ph_left = phasemaps['left'].copy()
    ph_right = phasemaps['right'].copy()
    ph_top = phasemaps['top'].copy()
    ph_bottom = phasemaps['bottom'].copy()
    if verbose:
        print("got phase:", np.nanmin(ph_left), np.nanmax(ph_left)) # (0, 2*np.pi)

    absolute_az = (ph_left - ph_right) / 2.
    delay_az = (ph_left + ph_right) / 2.
    absolute_el = (ph_bottom - ph_top) / 2.
    delay_el = (ph_bottom + ph_top) / 2.

    # Reset range
    vmin, vmax = (-np.pi, np.pi) 
    if verbose:
        print("got absolute:", np.nanmin(absolute_az), np.nanmax(absolute_az))
        print("Delay:", np.nanmin(delay_az), np.nanmax(delay_az))

    return magmaps, absolute_az, absolute_el, delay_az, delay_el

def plot_phase_and_delay_maps(absolute_az, absolute_el, delay_az, delay_el, 
                                cmap='nipy_spectral', vmin=-np.pi, vmax=np.pi, 
                                elev_cutoff=0.56):
    if cmap=='nic_Edge':
        screen, cmap = get_retino_legends(cmap_name=cmap, zero_center=True, 
                                                  return_cmap=True)

    abs_vmin, abs_vmax = (-np.pi, np.pi)
    del_vmin, del_vmax = (0, 2.*np.pi)

    fig, axes = pl.subplots(2,2)
    az_mean = spstats.circmean(absolute_az[~np.isnan(absolute_az)], low=abs_vmin, high=abs_vmax)
    az_std = spstats.circstd(absolute_az[~np.isnan(absolute_az)], low=abs_vmin, high=abs_vmax)
    im1 = axes[0,0].imshow(absolute_az, cmap=cmap, vmin=abs_vmin, vmax=abs_vmax)
    axes[0,0].set_title('Azimuth', fontsize=12, loc='left')
    axes[0,0].set_title('mean AZ %.2f (+/- %.2f)' % (az_mean, az_std), fontsize=8, loc='left')
    pplot.colorbar(im1)

    el_mean = spstats.circmean(absolute_el[~np.isnan(absolute_el)], low=abs_vmin, high=abs_vmax)
    el_std = spstats.circstd(absolute_el[~np.isnan(absolute_el)], low=abs_vmin, high=abs_vmax)
    im2 = axes[0,1].imshow(absolute_el, cmap=cmap, vmin=abs_vmin, vmax=abs_vmax)
    axes[0,1].set_title('mean EL %.2f (+/- %.2f)' % (el_mean, el_std), fontsize=8, loc='left')
    pplot.colorbar(im2)

    # Print some info to plot
    d_az_mean = spstats.circmean(delay_az[~np.isnan(delay_az)], low=del_vmin, high=del_vmax)
    d_az_std = spstats.circstd(delay_az[~np.isnan(delay_az)], low=del_vmin, high=del_vmax)
    im1b=axes[1,0].imshow(delay_az, cmap=cmap, vmin=del_vmin, vmax=del_vmax)
    axes[1,0].set_title('mean del %.2f (+/- %.2f)' % (d_az_mean, d_az_std),
                        loc='left', fontsize=8)
    pplot.colorbar(im1b)

    d_el_mean = spstats.circmean(delay_el[~np.isnan(delay_el)], low=del_vmin, high=del_vmax)
    d_el_std = spstats.circstd(delay_el[~np.isnan(delay_el)], low=del_vmin, high=del_vmax)
    im2b=axes[1,1].imshow(delay_el, cmap=cmap, vmin=del_vmin, vmax=del_vmax)
    axes[1,1].set_title('mean del %.2f (+/- %.2f)' % (d_el_mean, d_el_std),
                        loc='left', fontsize=8)
    pplot.colorbar(im2b)

    cbar1_orientation='horizontal'
    cbar1_axes = [0.35, 0.85, 0.12, 0.12]

    cbaxes = fig.add_axes(cbar1_axes) 
    cb = pl.colorbar(im1, cax = cbaxes, orientation=cbar1_orientation)  
    cb.ax.axis('off')
    cb.outline.set_visible(False)

    cbar2_orientation='vertical'
    cbar2_axes = [0.75, 0.85, 0.12, 0.12]
    cbaxes = fig.add_axes(cbar2_axes) 
    cb = pl.colorbar(im2, cax = cbaxes, orientation=cbar2_orientation)
    #cb.ax.set_ylim([cb.norm(-np.pi*top_cutoff), cb.norm(np.pi*top_cutoff)])
    #cb.ax.axhline(y=cb.norm(vmin*elev_cutoff), color='w', lw=1)
    #cb.ax.axhline(y=cb.norm(vmax*elev_cutoff), color='w', lw=1)
    cb.ax.axhline(y=vmin*elev_cutoff, color='k', lw=2, ls=':')
    cb.ax.axhline(y=vmax*elev_cutoff, color='k', lw=2, ls=':') 
    cb.ax.axis('off')
    cb.outline.set_visible(False)
    pl.subplots_adjust(top=0.8, hspace=0.5, wspace=0.5)

    for ax in axes.flat:
        ax.axis('off')
 
    return fig




def filter_by_delay_map(absolute_az, absolute_el, delay_az, delay_el, 
                        delay_map_thr=0.5, return_delay=True):
    '''
    Good pixels/cells should have delay close to 0 on absolute maps.
    Keep delay_map_thr low to filter harder.
    '''
    if delay_map_thr is None:
        delay_map_thr = np.inf # Set to inf. if None (no filter)

    delay_diff = abs(delay_az-delay_el)
    filt_az = absolute_az.copy()
    filt_az[delay_diff>delay_map_thr] = np.nan

    filt_el = absolute_el.copy()
    filt_el[delay_diff>delay_map_thr] = np.nan

    delay_filt = delay_diff.copy()
    delay_filt[delay_diff>delay_map_thr] = np.nan

    if return_delay:
        return filt_az, filt_el, delay_diff
    else:
        return filt_az, filt_el
    

# -----------------------------------------------------------------------------
# Image transformation/rotation to match WF
# -----------------------------------------------------------------------------
def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def transform_2p_fov(img, pixel_size, zoom_factor=1., normalize=True):
    '''
    First, left/right reflection and rotation of 2p image to match orientation of widefield view.
    Then, scale image to pixel size as measured by PSF.
    '''
    transf_ = orient_2p_to_macro(img, zoom_factor=zoom_factor, save=False, normalize=normalize)
    scaled_ = scale_2p_fov(transf_, pixel_size=pixel_size)
    return scaled_

def orient_2p_to_macro(avg, zoom_factor, normalize=True,
                    acquisition_dir='/tmp', channel_ix=0, plot=False, save=True): #,
                        #xaxis_conversion=2.312, yaxis_conversion=1.904):
    '''
    Does standard Fiji steps:
        1. Scale slow-angle (if needed)
        2. Rotate image leftward, and flip L/R ('horizontally' in Fiji)
        3. Convert to 8-bit and adjust contrast
    '''
    # Scale:
    d1, d2 = avg.shape # (img height, img width)
    
    # dsize: (v1, v2) -- v1 specifies COLS, v2 specifies ROWS (i.e., img_w, img_h)
    scaled = cv2.resize(avg, dsize=(int(d1*zoom_factor), d2), interpolation=cv2.INTER_CUBIC)  #, dtype=avg.dtype)

    # Rotate leftward:
    rotated = rotate_image(scaled, 90)

    # Flip L/R:
    transformed = np.fliplr(rotated)

    # Cut off super low vals, Convert range from 0, 255
    if normalize:
        transformed[transformed<-50] = 0
        normed = cv2.normalize(transformed, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to 8-bit
        img8 = cv2.convertScaleAbs(normed)

        # Equalize hist:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
        eq = clahe.apply(img8)
    else:
        eq = transformed.copy()

    if normalize:
        return img8 #eq #transformed_img_path
    else:
        return eq
    
def scale_2p_fov(transformed_image, pixel_size=(2.312, 1.888)):
    xaxis_conversion, yaxis_conversion = pixel_size

    d1, d2 = transformed_image.shape # d1=HEIGHT, d2=WIDTH
    new_d1 = int(round(d1*xaxis_conversion,1)) # yaxis corresponds to M-L axis (now along )
    new_d2 = int(round(d2*yaxis_conversion,1)) # xaxis corresopnds to A-P axis (d2 is iamge width)
    im_r = cv2.resize(transformed_image, (new_d2, new_d1))

    return im_r


# -----------------------------------------------------------------------------
# MW/protocol loading
# -----------------------------------------------------------------------------

def load_2p_surface(datakey, ch_num=1, retinorun='retino_run1', 
                    rootdir='/n/coxfs01/2p-data'):
    from skimage.measure import block_reduce
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    run_dir = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i_*' % fovn, 
                        retinorun))[0]
    fov_imgs = glob.glob(os.path.join(run_dir, 'processed', 'processed*', 
                                'mcorrected_*mean_deinterleaved',\
                                'Channel%02d' % ch_num, 'File*', '*.tif')) 
    imlist = []
    for anat in fov_imgs:
        im = tf.imread(anat)
        imlist.append(im)
    surface_img = np.array(imlist).mean(axis=0)
    
    return surface_img


def load_fov_image(RETID):

    ds_factor = int(RETID['PARAMS']['downsample_factor'])

    # Load reference image
    imgs = glob.glob(os.path.join('%s*' % RETID['SRC'], 'std_images.tif'))[0]
    #imgs = glob.glob(os.path.join(rootdir, animalid, session, fov, retinorun, 'processed',\
    #                      'processed001*', 'mcorrected_*', 'std_images.tif'))[0]
    zimg = tf.imread(imgs)
    zimg = zimg.mean(axis=0)

    if ds_factor is not None:
        zimg = block_mean(zimg, int(ds_factor))

    print("... FOV size: %s (downsample factor=%i)" % (str(zimg.shape), ds_factor))

    return zimg

def load_mw_info(datakey, run_name='retino', rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    parsed_fpaths = glob.glob(os.path.join(rootdir, animalid, session, 
                                'FOV%i_*' % fovn, '%s*' % run_name,
                                'paradigm', 'files', 'parsed_trials*.json'))
    assert len(parsed_fpaths)==1, "Unable to find correct parsed trials path: %s" % str(parsed_fpaths)
    with open(parsed_fpaths[0], 'r') as f:
        mwinfo = json.load(f)

    return mwinfo


def get_protocol_info(datakey, run='retino_run1', rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    run_dir = glob.glob(os.path.join(rootdir, animalid, session, 
                                    'FOV%i_*' % fovn, '%s*' % run))[0]
    mwinfo = load_mw_info(datakey, run_name=run, rootdir=rootdir)

    si_fpath = glob.glob(os.path.join(run_dir, '*.json'))[0]
    with open(si_fpath, 'r') as f:
        scaninfo = json.load(f)

    conditions = list(set([cdict['stimuli']['stimulus'] \
                        for trial_num, cdict in mwinfo.items()]))
    trials_by_cond = dict((cond, [int(k) for k, v in mwinfo.items() \
                        if v['stimuli']['stimulus']==cond]) \
                           for cond in conditions)
    n_frames = scaninfo['nvolumes']
    fr = scaninfo['frame_rate']

    stiminfo = dict((cond, dict()) for cond in conditions)
    curr_cond = conditions[0]
    # get some info from paradigm and run file
    stimfreq = np.unique([v['stimuli']['scale'] for k,v in mwinfo.items() if v['stimuli']['stimulus']==curr_cond])[0]
    stimperiod = 1./stimfreq # sec per cycle

    n_cycles = int(round((n_frames/fr) / stimperiod))
    n_frames_per_cycle = int(np.floor(stimperiod * fr))
    cycle_starts = np.round(np.arange(0, n_frames_per_cycle * n_cycles, n_frames_per_cycle)).astype('int')

    # Get frequency info
    freqs = np.fft.fftfreq(n_frames, float(1./fr))
    sorted_idxs = np.argsort(freqs)
    freqs = freqs[sorted_idxs] # sorted
    freqs = freqs[int(np.round(n_frames/2.))+1:] # exclude DC offset from data
    stim_freq_idx = np.argmin(np.absolute(freqs - stimfreq)) # Index of stimulation frequency

    stiminfo = {'stim_freq': stimfreq,
               'frame_rate': fr,
               'n_reps': len(trials_by_cond[curr_cond]),
               'n_frames': n_frames,
               'n_cycles': n_cycles,
               'n_frames_per_cycle': n_frames_per_cycle,
               'cycle_start_ixs': cycle_starts,
               'stim_freq_idx': stim_freq_idx,
               'freqs': freqs}

    scaninfo.update({'stimulus': stiminfo})
    scaninfo.update({'trials': trials_by_cond})


    return scaninfo

def load_retino_analysis_info(datakey, run='retino', roiid=None, retinoid=None, 
                              use_pixels=False, rootdir='/n/coxfs01/2p-data'):

    session, animalid, fovn = hutils.split_datakey_str(datakey)
    fov='FOV%i_zoom2p0x' % fovn

    retinoid=None; RID=None;
    run_dir = glob.glob(os.path.join(rootdir, animalid, session, \
                        'FOV%i_*' % fovn, '%s*' % run))[0]
    fov = os.path.split(os.path.split(run_dir)[0])[-1]
    try:
        retinoids_fpath = glob.glob(os.path.join(run_dir, 'retino_analysis', \
                                        'analysisids_*.json'))[0]
        with open(retinoids_fpath, 'r') as f:
            rids = json.load(f)
       
        if use_pixels:
            roi_analyses = [r for r, rinfo in rids.items() \
                            if rinfo['PARAMS']['roi_type'] == 'pixels']
        else:
            if roiid is not None:
                roi_analyses = [r for r, rinfo in rids.items() \
                            if 'roi_id' in rinfo['PARAMS'].keys() \
                            and rinfo['PARAMS']['roi_id']==roiid]
            else: 
                roi_analyses = [r for r, rinfo in rids.items() \
                            if rinfo['PARAMS']['roi_type'] != 'pixels']
        if retinoid not in roi_analyses:
            # use most recent roi analysis
            retinoid = sorted(roi_analyses, key=hutils.natural_keys)[-1]
        RID = rids[retinoid]

    except Exception as e:
        print(e)
        
    return retinoid, RID

# -----------------------------------------------------------------------------
# FFT
# -----------------------------------------------------------------------------
# FFT
def fft_results_by_trial(RETID):

    run_dir = RETID['DST'].split('/retino_analysis/')[0]
    processed_filepaths = glob.glob(os.path.join(RETID['DST'], 'files', '*h5'))
    trialinfo_filepath = glob.glob(os.path.join(run_dir, 'paradigm',
                                    'files', 'parsed_trials*.json'))[0]
    _, magratio, phase, trials_by_cond = trials_to_dataframes(processed_filepaths,
                                                trialinfo_filepath)
    return magratio, phase, trials_by_cond


def trials_to_dataframes(processed_fpaths, conditions_fpath):
    '''
    Open outupt files (from do_retinoanalysis.py) and return 
    processed data for each trial and condition
    Works for pixels or for ROIs.
    '''
    # Get condition / trial info:
    with open(conditions_fpath, 'r') as f:
        conds = json.load(f)
    cond_list = list(set([cond_dict['stimuli']['stimulus'] \
                    for trial_num, cond_dict in conds.items()]))
    trials_by_cond = dict((cond, [int(k) for k, v in conds.items() \
                    if v['stimuli']['stimulus']==cond]) for cond in cond_list)

    excluded_tifs = []
    for cond, tif_list in trials_by_cond.items():
        for tifnum in tif_list:
            processed_tif = [f for f in processed_fpaths \
                                if 'File%03d' % tifnum in f]
            if len(processed_tif) == 0:
                print("Warning (%s) No analysis found for file: %s" \
                            % (conditions_fpath, tifnum))
                excluded_tifs.append(tifnum)
        trials_by_cond[cond] = [t for t in tif_list if t not in excluded_tifs]
    trial_list = [int(t) for t in conds.keys() if int(t) not in excluded_tifs]

    fits = []
    phases = []
    mags = []
    for trial_num, trial_fpath \
        in zip(sorted(trial_list), sorted(processed_fpaths, key=hutils.natural_keys)):
        #print("%i: %s" % (trial_num, os.path.split(trial_fpath)[-1]))
        df = h5py.File(trial_fpath, 'r')
        fits.append(pd.Series(data=df['var_exp_array'][:], name=trial_num))
        phases.append(pd.Series(data=df['phase_array'][:], name=trial_num))
        mags.append(pd.Series(data=df['mag_ratio_array'][:], name=trial_num))
        df.close()

    fit = pd.concat(fits, axis=1)
    magratio = pd.concat(mags, axis=1)
    phase = pd.concat(phases, axis=1)

    return fit, magratio, phase, trials_by_cond


def extract_from_fft_results(fft_soma):
    '''
    Return magratios_soma, phases_soma as dataframe
    '''
    # Create dataframe of magratios -- each column is a condition
    magratios_soma = pd.DataFrame(dict((cond, k[0]) for cond, k in fft_soma.items()))
    phases_soma = pd.DataFrame(dict((cond, k[1]) for cond, k in fft_soma.items()))

    return magratios_soma, phases_soma


def load_fft_results(datakey, retinorun='retino_run1', trace_type='corrected',
                     traceid='traces001', roiid=None, use_pixels=False,
                     detrend_after_average=True, in_negative=False,
                     rootdir='/n/coxfs01/2p-data', verbose=False, create_new=False):
    
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    fft_results=None
    run_dir = glob.glob(os.path.join(rootdir, animalid, session, \
                            'FOV%i_*' % fovn, retinorun))
    try:
        # RETID = load_retinoanalysis(run_dir, traceid)
        retinoid, RETID = load_retino_analysis_info(datakey, 
                                    use_pixels=use_pixels, roiid=roiid)

        assert RETID is not None
    except AssertionError as e: #Exception as e:
        print("[%s] NO retino <%s> for rois w/ %s\n...check dir: %s" \
                    % (datakey, retinorun, traceid, run_dir))
        #create_new=True
        return None

    analysis_dir = RETID['DST']
    retinoid = RETID['analysis_id']
    if verbose:
        print("... Loaded: %s, %s (%s))" % (retinorun, retinoid, run_dir))

    fft_dpath=os.path.join(analysis_dir, 'fft_results_%s.pkl' % retinoid)
    if create_new is False:
        try:
            with open(fft_dpath, 'rb') as f:
                fft_results = pkl.load(f)
        except Exception as e:
            create_new=True

    if create_new:
        # Load MW info and SI info
        mwinfo = load_mw_info(datakey, retinorun)
        scaninfo = get_protocol_info(datakey, run=retinorun) # load_si(run_dir)
        tiff_paths = sorted(glob.glob(os.path.join(RETID['SRC'], '*.tif')), 
                                key=hutils.natural_keys)
        if verbose:
            print("Found %i tifs" % len(tiff_paths))

        # Some preprocessing params
        temporal_ds = float(RETID['PARAMS']['average_frames'])
        if verbose:
            print("Temporal ds: %.2f" % (temporal_ds))

        #### Load raw and process traces -- returns average trace for condition
        # retino_dpath = os.path.join(analysis_dir, 'traces', 'extracted_traces.h5')
        np_traces = load_roi_traces(datakey, run=retinorun,
                            analysisid=retinoid, trace_type='neuropil', 
                            detrend_after_average=detrend_after_average)
        soma_traces = load_roi_traces(datakey, run=retinorun,
                            analysisid=retinoid, trace_type=trace_type, 
                            detrend_after_average=detrend_after_average)
        # Do fft
        n_frames = scaninfo['stimulus']['n_frames']
        frame_rate = scaninfo['stimulus']['frame_rate']
        stim_freq_idx = scaninfo['stimulus']['stim_freq_idx']

        #### label frequency bins
        freqs = np.fft.fftfreq(n_frames, float(1./frame_rate))
        sorted_freq_idxs = np.argsort(freqs)

        sign = -1 if in_negative else 1
        
        fft_soma = dict((cond, do_fft_analysis(sign*tdf, sorted_freq_idxs, stim_freq_idx)) \
                        for cond, tdf in soma_traces.items())
        fft_np=None
        if use_pixels is False:
            fft_np = dict((cond, do_fft_analysis(tdf, sorted_freq_idxs, stim_freq_idx)) \
                      for cond, tdf in np_traces.items())

        fft_results = {'fft_soma': fft_soma, 'fft_neuropil': fft_np,
                       'scaninfo': scaninfo, 'RETID': RETID}

        #### Save output
        with open(fft_dpath, 'wb') as f:
            pkl.dump(fft_results, f, protocol=2)

    return fft_results


def get_retino_fft(datakey, curr_cells=None, traceid='traces001', 
                   mag_thr=None, delay_thr=None, create_new=False,
                   use_pixels=False):
    '''
    Get retino phase/mag for AZ and EL by datakey and assigned cells.
    Set mag_thr to super low # if don't want to threshold yet.
    If >1 retinorun, pick the best one (based on max. mag-ratio).
    Set mag_thr=None, delay_thr=None for NO filters.
    '''
    df=None
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    fov='FOV%i_zoom2p0x' % fovn
    try:
        roiid = roiutils.get_roiid_from_traceid(animalid, session, fov, 
                                                traceid=traceid)
    except Exception as e:
        print("[%s] Unable to get roiid (%s)" % (datakey, traceid))
        print(e)
        roiid=None
    # Select best retino run (if there are multiple)
    all_retinos = get_average_mag_across_pixels(datakey)
    try:
        retinorun = all_retinos.loc[all_retinos[1].idxmax()][0]
    except Exception as e:
        print(e)
        print(all_retinos)
    # Load fft results
    fft_results = load_fft_results(datakey, roiid=roiid,
                                    retinorun=retinorun, traceid=traceid, 
                                    create_new=create_new, 
                                    use_pixels=use_pixels)
    if fft_results is None:
        return None

    fft_soma = fft_results['fft_soma']
    # Create dataframe of magratios -- each column is a condition
    conds=['left', 'right', 'bottom', 'top']
    df_=None
    try:
        magratios_soma, phases_soma = extract_from_fft_results(fft_soma)
        assert all([a in magratios_soma.columns for a in conds]), \
                "Incorrect N conditions (%s)" % str(magratios_soma.columns)
        # Get maps
        df_ = get_final_maps(magratios_soma, phases_soma, 
                        trials_by_cond=None, mag_thr=mag_thr, 
                        delay_thr=delay_thr)
        # Select cells
        if df_ is not None:
            if curr_cells is None:
                cell_cells = df.index.to_numpy()
            cell_ids = curr_cells['cell'].unique()
            df = df_.loc[cell_ids].copy()
            df['retinorun'] = retinorun
    except Exception as e:
        print(e)
        return None

    return df


def do_fft_analysis(avg_traces, sorted_idxs, stim_freq_idx):
    n_frames = avg_traces.shape[0]

    fft_results = np.fft.fft(avg_traces, axis=0) #avg_traces.apply(np.fft.fft, axis=1)

    # get phase and magnitude
    mag_data = abs(fft_results)
    phase_data = np.angle(fft_results)

    # sort mag and phase by freq idx:
    mag_data = mag_data[sorted_idxs]
    phase_data = phase_data[sorted_idxs]

    # exclude DC offset from data
    if len(mag_data.shape)==1:
        mag_data = mag_data[int(np.round(n_frames/2.))+1:]
        phase_data = phase_data[int(np.round(n_frames/2.))+1:]
        #unpack values from frequency analysis
        mag_array = mag_data[stim_freq_idx]
        phase_array = phase_data[stim_freq_idx]
    else:
        mag_data = mag_data[int(np.round(n_frames/2.))+1:, :]
        phase_data = phase_data[int(np.round(n_frames/2.))+1:, :]

        #unpack values from frequency analysis
        mag_array = mag_data[stim_freq_idx, :]
        phase_array = phase_data[stim_freq_idx, :]

    #get magnitude ratio
    tmp = np.copy(mag_data)
    #tmp = np.delete(tmp,freq_idx,0)
    nontarget_mag_array=np.sum(tmp,0)
    magratio_array=mag_array/nontarget_mag_array

    return magratio_array, phase_array


# -----------------------------------------------------------------------------
# Traces/preprocessing
# -----------------------------------------------------------------------------
# preprocessing ---------------
def load_roi_traces(datakey, run='retino_run1', analysisid='analysis002',
                trace_type='corrected', 
                detrend_after_average=True, temporal_ds=None,
                verbose=False,
                rootdir='/n/coxfs01/2p-data'):
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    if verbose:
        print("... loading traces (%s)" % trace_type)
    retinoid_path = glob.glob(os.path.join(rootdir, animalid, session, 
                           'FOV%i_*' % fovn, '%s*' % run,
                            'retino_analysis', 'analysisids_*.json'))[0]
    with open(retinoid_path, 'r') as f:
        RIDS = json.load(f)
    eligible = [r for r, res in RIDS.items() if res['PARAMS']['roi_type']!='pixels']
    if analysisid not in eligible:
        print("Specified ID <%s> not eligible. Selecting 1st of %s"
                    % (analysisid, str(eligible)))
        analysisid = eligible[0]

    analysis_dir = RIDS[analysisid]['DST']
    if verbose:
        print("... loading traces from: %s" % analysis_dir)
    retino_dpath = os.path.join(analysis_dir, 'traces', 'extracted_traces.h5')
    scaninfo = get_protocol_info(datakey, run=run)
    if temporal_ds is None:
        temporal_ds = RIDS[analysisid]['PARAMS']['downsample_factor']
    traces = load_roi_traces_from_file(retino_dpath, scaninfo, trace_type=trace_type,
                                    temporal_ds=temporal_ds, 
                                    detrend_after_average=detrend_after_average)

    return traces


def load_roi_traces_from_file(retino_dpath, scaninfo, trace_type='corrected',
                            temporal_ds=None, detrend_after_average=False):
    '''
    Pre-processes raw extracted traces by:
        - adding back in neuropil offsets, and
        - F0 offset from drift correction.
    Loads: ./traces/extracted_traces.h5 (contains data for each tif file).
    Averages traces for each condition. Downsamples final array.
    '''
    frame_rate = scaninfo['stimulus']['frame_rate']
    stim_freq = scaninfo['stimulus']['stim_freq']
    trials_by_cond = scaninfo['trials']

    traces = {}
    try:
        tfile = h5py.File(retino_dpath, 'r')
        for condition, trialnums in trials_by_cond.items():
            #print("... loading cond: %s" % condition)
            do_detrend = detrend_after_average is False
            dlist = tuple([process_data(tfile, trialnum, trace_type=trace_type, \
                        frame_rate=frame_rate, stim_freq=stim_freq, detrend=do_detrend) \
                        for trialnum in trialnums])
            dfcat = pd.concat(dlist)
            df_rowix = dfcat.groupby(dfcat.index)
            # Average raw traces together
            meandf = df_rowix.mean()

            # detrend
            if detrend_after_average:
                f0 = meandf.mean() #.mean()
                drift_corr = detrend_array(meandf, frame_rate=frame_rate, stim_freq=stim_freq)
                meandf = drift_corr + f0

            # smooth
            if temporal_ds is not None:
                #print("Temporal ds: %.2f" % temporal_ds)
                meandf = downsample_array(meandf, temporal_ds=temporal_ds)
            traces[condition] = meandf
    except Exception as e:
        traceback.print_exc()
    finally:
        tfile.close()

    return traces

def process_data(tfile, trialnum, trace_type='corrected', add_offset=True, #'corrected', add_offset=True,
                frame_rate=44.65, stim_freq=0.13, correction_factor=0.7, detrend=True):
    #print(tfile['File001'].keys())
    if trace_type != 'neuropil' and add_offset:
        # Get raw soma traces and raw neuropil -- add neuropil offset to soma traces
        # print(tfile['File%03d' % int(trialnum)].keys())
        # trace_types:  corrected, neuropil, processed, raw (+ masks)
        soma = pd.DataFrame(tfile['File%03d' % int(trialnum)][trace_type][:].T)
        neuropil = pd.DataFrame(tfile['File%03d' % int(trialnum)]['neuropil'][:].T)
        np_offset = neuropil.mean(axis=0) #neuropil.mean().mean()
        if trace_type=='raw':
            #print("raw")
            xd = soma.subtract(correction_factor*neuropil) + np_offset
        else:
            xd = soma + np_offset
        del neuropil
        del soma
    else:
        xd = pd.DataFrame(tfile['File%03d' % int(trialnum)][trace_type][:].T)

    if detrend:
        f0 = xd.mean() #.mean()
        drift_corrected = detrend_array(xd, frame_rate=frame_rate, stim_freq=stim_freq)
        xdata = drift_corrected + f0
    else:
        xdata = xd.copy()
    #if temporal_ds is not None:
    #    xdata = downsample_array(xdata, temporal_ds=temporal_ds)

    return xdata

def subtract_rolling_mean(trace, windowsz):
    #print(trace.shape)
    tmp1 = np.concatenate((np.ones(windowsz)*trace.values[0], trace, np.ones(windowsz)*trace.values[-1]),0)
    rolling_mean = np.convolve(tmp1, np.ones(windowsz)/windowsz, 'same')
    rolling_mean=rolling_mean[windowsz:-windowsz]
    return np.subtract(trace, rolling_mean)

def detrend_array(roi_trace, frame_rate=44.65, stim_freq=0.24):
    #print('Removing rolling mean from traces...')
    windowsz = int(np.ceil((np.true_divide(1,stim_freq)*3)*frame_rate))
    detrend_roi_trace = roi_trace.apply(subtract_rolling_mean, args=(windowsz,), axis=0)
    return detrend_roi_trace #pd.DataFrame(detrend_roi_trace)

def temporal_downsample(trace, windowsz):
    tmp1=np.concatenate((np.ones(windowsz)*trace.values[0], trace, np.ones(windowsz)*trace.values[-1]),0)
    tmp2=np.convolve(tmp1, np.ones(windowsz)/windowsz, 'same')
    tmp2=tmp2[windowsz:-windowsz]
    return tmp2


def downsample_array(roi_trace, temporal_ds=5):
    #print('Performing temporal smoothing on traces...')
    windowsz = int(temporal_ds)
    smooth_roi_trace = roi_trace.apply(temporal_downsample, args=(windowsz,), axis=0)
    return smooth_roi_trace


# -----------------------------------------------------------------------------
# ROIs/Mask manipulation + dilation
# -----------------------------------------------------------------------------
from scipy.ndimage.morphology import binary_dilation
from scipy import ndimage

def get_kernel(kernel_size):
    kernel_radius = (kernel_size - 1) // 2
    x, y = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    dist = (x**2 + y**2)**0.5 # shape (kernel_size, kernel_size)

    # let's create three kernels for the sake of example
    radii = np.array([kernel_size/3., kernel_size/2.5, kernel_size/2.])[...,None,None] # shape (num_radii, 1, 1)
    # using ... allows compatibility with arbitrarily-shaped radius arrays

    kernel = (1 - (dist - radii).clip(0,1)).sum(axis=0)
    return kernel

def dilate_mask_centers(maskcenters, kernel_size=9):
    '''Calculate center of soma, then dilate to create masks for smoothed  neuropil
    '''
#     a = np.zeros((5, 5))
#     struct1 = ndimage.generate_binary_structure(2, 1)   
#     kernel = ndimage.binary_dilation(a, structure=struct1, iterations=sigma).astype(a.dtype)
    
    kernel = get_kernel(kernel_size)
    dilated_masks = np.zeros(maskcenters.shape, dtype=maskcenters.dtype)
    for roi in range(maskcenters.shape[0]):
        img = maskcenters[roi, :, :].copy()
        x, y = np.where(img>0)
        # calculate centroid
        centroid = ( int(round(sum(x) / len(x))), int(round(sum(y) / len(x))) )
        # print(centroid)
        np_tmp = np.zeros(img.shape, dtype=bool)
        np_tmp[centroid] = True
        dilation = binary_dilation(np_tmp, structure=kernel )
        dilated_masks[roi, : :] = dilation
    return dilated_masks


def dilate_from_centroids(masks_r, kernel_size=9):
    '''Calculate center of soma, then dilate to create masks for smoothed  neuropil
    '''
    # Load centroids
    centroids = roiutils.calculate_roi_centroids(masks_r, xlabel='ml_pos', ylabel='ap_pos')
    centroids = centroids.astype(int)
    # Create masks from them
    np_tmp = np.zeros(masks_r.shape, dtype=bool)
    for i in range(centroids.shape[0]):
        np_tmp[i, centroids['ap_pos'].iloc[i], centroids['ml_pos'].iloc[i]] = True
    # Dilate
    kernel = get_kernel(kernel_size)
    dilated_masks = np.array([binary_dilation(np_tmp[i, :], structure=kernel ) 
          for i in range(np_tmp.shape[0])])

    return dilated_masks, centroids


def dilate_centroids(datakey, desired_radius_um=20, traceid='traces001',
                    xlabel='ml_pos', ylabel='ap_pos'):   
    # Get pixel size
    pixel_size = hutils.get_pixel_size()
    um_per_pixel = np.mean(pixel_size)
    pixels2dilate = desired_radius_um/um_per_pixel
    kernel_size = np.ceil(pixels2dilate+2) #21
    kernel = get_kernel(kernel_size)

    # Load masks and centroids
    zimg, masks, ctrs = roiutils.get_masks_and_centroids(datakey, traceid=traceid,
                                    xlabel=xlabel, ylabel=ylabel)
    centroids = ctrs.astype(int)
    # Create masks from them
    np_tmp = np.zeros(masks.shape, dtype=bool)
    for i in range(centroids.shape[0]):
        np_tmp[i, centroids[ylabel].iloc[i], centroids[xlabel].iloc[i]] = True

    # Dilate
    dilated_masks = np.array([binary_dilation(np_tmp[i, :], structure=kernel ) 
          for i in range(np_tmp.shape[0])])

    return zimg, dilated_masks, centroids

 
 

#def get_roi_centroids(masks):
#    '''Calculate center of soma, then return centroid coords.
#    '''
#    centroids=[]
#    for roi in range(masks.shape[0]):
#        img = masks[roi, :, :].copy()
#        y, x = np.where(img>0)
#        centroid = ( round(sum(x) / len(x)), round(sum(y) / len(x)) )
#        centroids.append(centroid)
#    
#    nrois_total = masks.shape[0]
#    ctr_df = pd.DataFrame(centroids, columns=['x', 'y'], index=range(nrois_total))
#
#    return ctr_df
#
def mask_rois(masks, value_array, mask_thr=0.1, return_array=False):
    '''
    dim 0 of mask_array (nrois, d1, d2) must be same as len(value_array)
    OR, value_array should be a dataframe with included rois, used to index into full masks array
    Indices of value_array (dataframe) should correspond actual index in masks.
    '''
    nr, d1_, d2_ = masks.shape
    dims = (d1_, d2_)

    if return_array:
        value_mask = np.ones(masks.shape)*np.nan #-100
        for rid in value_array.index.tolist():
            value_mask[rid, masks[rid,:,:]>=mask_thr] = value_array.loc[rid]

    else:
        value_mask =  np.ones(dims)*-100
        for rid in value_array.index.tolist():
            value_mask[masks[rid,:,:]>=mask_thr] = value_array.loc[rid]

    return value_mask

def mask_with_overlaps_averaged(dilated_masks, value_array, mask_thr=0.1):
    '''
    Get 2D roi mask, with overlapping masks averaged
    '''
    nt, d1_, d2_ = dilated_masks.shape
    
    # Get non-averaged array
    tmpmask = mask_rois(dilated_masks, value_array, mask_thr=mask_thr, return_array=False)
    
    # Get full array to average across overlapping pixels
    tmpmask_full = mask_rois(dilated_masks, value_array, \
                                mask_thr=mask_thr, return_array=True)
    tmpmask_r = np.reshape(tmpmask_full, (nt, d1_*d2_))
    
    # Replace overlapping pixels with average value
    avg_mask = tmpmask.copy().ravel()
    multi_ixs = [i for i in range(tmpmask_r.shape[-1]) \
                    if len(np.where(tmpmask_r[:, i])[0]) > 1]
    for ix in multi_ixs:
        #avg_azim[ix] = spstats.circmean([v for v in azim_phase2[:, ix] if not np.isnan(v)], low=vmin, high=vmax)
        avg_mask[ix] = np.nanmean([v for v in tmpmask_r[:, ix] if not np.isnan(v)])#, low=vmin, high=vmax)

    avg_mask = np.reshape(avg_mask, (d1_, d2_))

    return avg_mask


def get_phase_masks(masks, azim, elev, average_overlap=True, roi_list=None): #, use_cont=True, mask_thr=0.01):
#     # Convert phase to continuous:
#     phases_cont = -1 * phases
#     phases_cont = phases_cont % (2*np.pi)
   
    '''
    masks: nrois, d1, d2 shape
    azim/elev:  datframes witih values to assign to roi masks (index should be actual roi ID used to index into masks array)
    ''' 
    # Only include specified rois:
    if roi_list is None:
        roi_list = azim.index.tolist()
        
#     # Get absolute maps:
#     if use_cont:
#         elev = (phases_cont['bottom'] - phases_cont['top']) / 2.
#         azim = (phases_cont['left'] - phases_cont['right']) / 2.
#         vmin = -np.pi
#         vmax = np.pi
#     else:
#         # Get absolute maps:
#         elev = (phases['bottom'] - phases['top']) / 2.
#         azim = (phases['left'] - phases['right']) / 2.
        
#         # Convert to continueous:
#         elev_c = -1 * elev
#         elev_c = elev_c % (2*np.pi)
#         azim_c = -1 * azim
#         azim_c = azim_c % (2*np.pi)

#         vmin = 0
#         vmax = 2*np.pi

#         azim = copy.copy(azim_c)
#         elev = copy.copy(elev_c)
        
    if average_overlap:
        azim_phase = mask_with_overlaps_averaged(masks, azim.loc[roi_list]) #, mask_thr=mask_thr)
        elev_phase = mask_with_overlaps_averaged(masks, elev.loc[roi_list]) #, mask_thr=mask_thr)
    else:
        azim_phase = mask_rois(masks, azim.loc[roi_list]) #, mask_thr=mask_thr)
        elev_phase = mask_rois(masks, elev.loc[roi_list]) #, mask_thr=mask_thr)   
    
    return azim_phase, elev_phase


# #########################################################################
# Widefield
# #########################################################################
# smoothing ------------------
def smooth_neuropil(azim_r, smooth_fwhm=21):
    V=azim_r.copy()
    V[np.isnan(azim_r)]=0
    VV=ndimage.gaussian_filter(V,sigma=smooth_fwhm)

    W=0*azim_r.copy()+1
    W[np.isnan(azim_r)]=0
    WW=ndimage.gaussian_filter(W,sigma=smooth_fwhm)

    azim_smoothed = VV/WW
    return azim_smoothed


def smooth_phase_nans(inputArray, sigma, sz):
    
    V=inputArray.copy()
    V[np.isnan(inputArray)]=0
    VV=smooth_phase_array(V,sigma,sz)

    W=0*inputArray.copy()+1
    W[np.isnan(inputArray)]=0
    WW=smooth_phase_array(W,sigma,sz)

    Z=VV/WW

    return Z

def smooth_array(inputArray, fwhm, phaseArray=False):
    szList=np.array([None,None,None,11,None,21,None,27,None,31,None,37,None,43,None,49,None,53,None,59,None,55,None,69,None,79,None,89,None,99])
    sigmaList=np.array([None,None,None,.9,None,1.7,None,2.6,None,3.4,None,4.3,None,5.1,None,6.4,None,6.8,None,7.6,None,8.5,None,9.4,None,10.3,None,11.2,None,12])
    sigma=sigmaList[fwhm]
    sz=szList[fwhm]
    #print(sigma, sz)
    if phaseArray:
        outputArray = smooth_phase_array(inputArray,sigma,sz)
    else:
        outputArray=cv2.GaussianBlur(inputArray, (sz,sz), sigma, sigma)
        
    return outputArray
        
def smooth_phase_array(theta,sigma,sz):
    #build 2D Gaussian Kernel
    kernelX = cv2.getGaussianKernel(sz, sigma); 
    kernelY = cv2.getGaussianKernel(sz, sigma); 
    kernelXY = kernelX * kernelY.transpose(); 
    kernelXY_norm=np.true_divide(kernelXY,np.max(kernelXY.flatten()))
    
    #get x and y components of unit-length vector
    componentX=np.cos(theta)
    componentY=np.sin(theta)
    
    #convolce
    componentX_smooth=signal.convolve2d(componentX,kernelXY_norm,mode='same',boundary='symm')
    componentY_smooth=signal.convolve2d(componentY,kernelXY_norm,mode='same',boundary='symm')

    theta_smooth=np.arctan2(componentY_smooth,componentX_smooth)
    return theta_smooth

def shift_map(phase_az):
    phaseC_az=np.copy(phase_az)
    phaseC_az[phase_az<0]=-phase_az[phase_az<0]
    phaseC_az[phase_az>0]=(2*np.pi)-phase_az[phase_az>0]

#    if phase_az[~np.isnan(phase_az)].min() < 0 and phase_az[~np.isnan(phase_az)].max() > 0:
#        phaseC_az[phase_az<0]=-phase_az[phase_az<0]
#        phaseC_az[phase_az>0]=(2*np.pi)-phase_az[phase_az>0]
#    else:
#        print("Already non-negative (min/max: %.2f, %.2f)" % (phase_az.min(), phase_az.max()))
    return phaseC_az


def convert_to_absolute(cond_data, smooth_fwhm=7, smooth=True, power_metric='mag'):
    '''combine absolute, or shift single-cond map so that
    
    if AZI, 0=left, 2*np.pi=right 
    if ALT, 0=bottom 2*np.pi= top
    
    Use this to convert to linear coords, centered around 0.
    power_metric: can be 'mag' or 'magRatio' (for Map type saved in analyzed maps).
    '''
    vmin = 0
    vmax = 2*np.pi

    combined_phase_map = convert_absolute_phasemap(cond_data, 
                                    smooth_fwhm=smooth_fwhm, smooth=smooth)
    combined_mag_map =  convert_absolute_magmap(cond_data, 
                                    smooth_fwhm=smooth_fwhm, smooth=smooth, 
                                                power_metric=power_metric)

    return combined_phase_map, combined_mag_map #_shift



def convert_absolute_phasemap(cond_data, smooth_fwhm=7, smooth=True):
    '''combine absolute, or shift single-cond map so that
    
    if AZI, 0=left, 2*np.pi=right 
    if ALT, 0=bottom 2*np.pi= top
    
    Use this to convert to linear coords, centered around 0.
    
    power_metric: can be 'mag' or 'magRatio' (for Map type saved in analyzed maps).
    
    '''
    vmin = 0
    vmax = 2*np.pi

    if len(cond_data.keys()) > 1:
        print("True absolute")
        c1 = 'left' if 'left' in cond_data.keys() else 'top'
        c2 = 'right' if c1=='left' else 'bottom'
    
        # Phase maps
        if smooth:
            m1 = shift_map(smooth_array(cond_data[c1]['phaseMap'], smooth_fwhm, phaseArray=True))
            m2 = shift_map(smooth_array(cond_data[c2]['phaseMap'], smooth_fwhm, phaseArray=True))
        else:
            m1 = shift_map(cond_data[c1]['phaseMap'])
            m2 = shift_map(cond_data[c2]['phaseMap'])
            
        combined_phase_map = spstats.circmean(np.dstack([m1, m2]), axis=-1, low=vmin, high=vmax) 

    else:
        print("Single cond")
        if 'right' in cond_data.keys() and 'top' not in cond_data.keys():
            m1 = cond_data['right']['phaseMap'].copy()
            m2 = cond_data['right']['phaseMap'].copy()*-1
            
        elif 'top' in cond_data.keys() and 'right' not in cond_data.keys():
            m1 = cond_data['top']['phaseMap'].copy()
            m2 = cond_data['top']['phaseMap'].copy()*-1
            
        # Phase maps
        combined_phase_map = (m2-m1)/2.
        
        if smooth:
            combined_phase_map = smooth_array(combined_phase_map, smooth_fwhm, phaseArray=True)
        
        # Shift maps
        combined_phase_map = shift_map(combined_phase_map) # values should go from 0 to 2*pi        
    
    return combined_phase_map #_shift


def convert_absolute_magmap(cond_data, smooth_fwhm=7, smooth=True, power_metric='mag'):
    '''combine absolute, or shift single-cond map so that
    
    if AZI, 0=left, 2*np.pi=right 
    if ALT, 0=bottom 2*np.pi= top
    
    Use this to convert to linear coords, centered around 0.
    
    power_metric: can be 'mag' or 'magRatio' (for Map type saved in analyzed maps).
    
    '''
    if len(cond_data.keys()) > 1:
        print("True absolute")
        c1 = 'left' if 'left' in cond_data.keys() else 'top'
        c2 = 'right' if c1=='left' else 'bottom'
        
        # Mag maps
        if smooth:
            p1 = smooth_array(cond_data[c1]['%sMap' % power_metric], smooth_fwhm, phaseArray=False)
            p2 = smooth_array(cond_data[c2]['%sMap' % power_metric], smooth_fwhm, phaseArray=False)
        else:
            p1 = cond_data[c1]['%sMap' % power_metric]
            p2 = cond_data[c2]['%sMap' % power_metric]
            
        combined_mag_map = np.mean(np.dstack([p1, p2]), axis=-1)

    else:
        print("Single cond")
        if 'right' in cond_data.keys() and 'top' not in cond_data.keys():
            p1 = cond_data['right']['%sMap' % power_metric].copy()
            
        elif 'top' in cond_data.keys() and 'right' not in cond_data.keys():
            p1 = cond_data['top']['%sMap' % power_metric].copy()
        
        # Mag maps
        combined_mag_map = p1
        
        if smooth:
            combined_mag_map = smooth_array(combined_mag_map, smooth_fwhm, phaseArray=False)
        
    return combined_mag_map #_shift


# PLOTTING
from matplotlib.colors import LinearSegmentedColormap

def create_legend(screen, zero_center=False):
    screen_x = screen['azimuth_deg']
    screen_y = screen['azimuth_deg'] #screen['altitude_deg']

    x = np.linspace(0, 2*np.pi, int(round(screen_x)))
    y = np.linspace(0, 2*np.pi, int(round(screen_y)) )
    xv, yv = np.meshgrid(x, y)

    az_legend = (2*np.pi) - xv
    el_legend = yv

    newmin = -0.5*screen_x if zero_center else 0
    newmax = 0.5*screen_x if zero_center else screen_x
    
    az_screen = hutils.convert_range(az_legend, newmin=newmin, newmax=newmax, 
                                oldmin=0, oldmax=2*np.pi)
    el_screen = hutils.convert_range(el_legend, newmin=newmin, newmax=newmax, 
                                oldmin=0, oldmax=2*np.pi)

    return az_screen, el_screen


def save_legend(az_screen, screen, cmap, cmap_name='cmap_name', cond='cond', dst_dir='/tmp'):
    screen_min = int(round(az_screen.min()))
    screen_max = int(round(az_screen.max()))
    #print("min/max:", screen_min, screen_max)
    
    fig, ax = pl.subplots()
    im = ax.imshow(az_screen, cmap=cmap)
    #ax.invert_xaxis()
   
    # Max value is twice the 0-centered value, or just the full value if not 0-cent
    max_v = screen['azimuth_deg'] #az_screen.max()*2.0 if screen_min < 0 else az_screen.max() #screen_max
  
    # Get actual screen edges
    midp = max_v/2.
    yedge_from_bottom = midp + screen['altitude_deg']/2.
    yedge_from_top = midp - screen['altitude_deg']/2.
    screen_edges_y = (-screen['altitude_deg']/2., screen['altitude_deg']/2.)

    if cond=='azimuth':
        ax.set_xticks(np.linspace(0, max_v, 5))
        ax.set_xticklabels([int(round(i)) for i \
                                in np.linspace(screen_min, screen_max, 5)][::-1])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis='x', length=0)
        ax.set_xlim(ax.get_xlim()[::-1])
    
    else:

        ax.set_yticks(np.linspace(0, min(az_screen.shape), 5))
        ax.set_yticklabels([int(round(i)) for i \
                                in np.linspace(screen_min, screen_max, 5)])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis='y', length=0)

        #ax.axhline(y=yedge_from_bottom, color='w', lw=2)
        #ax.axhline(y=yedge_from_top, color='w', lw=2)
        #print(screen_edges_y)
        ax.set_ylim(ax.get_ylim()[::-1])

    ax.axhline(y=yedge_from_bottom, color='w', lw=2)
    ax.axhline(y=yedge_from_top, color='w', lw=2)

    ax.set_frame_on(False)
    pl.colorbar(im, ax=ax, shrink=0.7)

    figname = '%s_pos_%s_LEGEND_abs' % (cond, cmap_name)
    pl.savefig(os.path.join(dst_dir, '%s.svg' % figname))

    print(dst_dir, figname)

    return


    
def make_legends(cmap='nipy_spectral', cmap_name='nipy_spectral', zero_center=False,
                 dst_dir='/n/coxfs01/julianarhee/aggregate-data/retinotopy'):

    screen = hutils.get_screen_dims()
    azi_legend, alt_legend = create_legend(screen, zero_center=zero_center)
   
    if dst_dir is not None:
        save_legend(azi_legend, screen, cmap=cmap, 
                        cmap_name=cmap_name, cond='azimuth', dst_dir=dst_dir)
        save_legend(alt_legend, screen, cmap=cmap, 
                        cmap_name=cmap_name, cond='elevation', dst_dir=dst_dir)
        
    screen.update({'azi_legend': azi_legend,
                   'alt_legend': alt_legend})
    return screen


def get_retino_legends(cmap_name='nic_edge', zero_center=True, return_cmap=False,
                    cmap_dir='/n/coxfs01/julianarhee/colormaps', 
                    dst_dir='/n/coxfs01/julianarhee/aggregate-visual-areas/retinotopy'):
    #colormap = 'nic_Edge'
    #cmapdir = os.path.join(aggr_dir, 'colormaps')
    cdata = np.loadtxt(os.path.join(cmap_dir, cmap_name) + ".txt")
    cmap_phase = LinearSegmentedColormap.from_list(cmap_name, cdata[::-1])
    screen = make_legends(cmap=cmap_phase, cmap_name=cmap_name, zero_center=zero_center,
                            dst_dir=dst_dir)
    if return_cmap:
        return screen, cmap_phase
    else:
        return screen
 
