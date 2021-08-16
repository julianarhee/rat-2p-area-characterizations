#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:56:01 2018

@author: julianarhee
"""
import os
import cv2
import traceback 
import h5py
import glob
import json
import imutils
import h5py
import tifffile as tf
import _pickle as pkl
import numpy as np
import pandas as pd

import analyze2p.utils as hutils

# --------------------------------------------------------------------
# Masks 
# --------------------------------------------------------------------
def masks_to_normed_array(masks):
    '''
    Assumes masks.shape = (d1, d2, nrois)

    Returns:
        maskarray of shape (d, nrois), where d = d1*d2
        values are normalized by size of mask
    '''
    d1, d2 = masks[:,:,0].shape
    d = d1*d2
    nrois = masks.shape[-1]
    masks_arr = np.empty((d, nrois))
    for rix in range(nrois):
        masks_arr[:, rix] = np.reshape(masks[:,:,rix], (d,), order='C') /  len(np.nonzero(masks[:,:,rix])[0])

    return masks_arr

# Loading
def get_mask_info(TID, RID, nslices=1, rootdir='/n/coxfs01/2p-data'):

    mask_path = os.path.join(RID['DST'], 'masks.hdf5')
    if rootdir not in mask_path and '/mnt/odyssey' in mask_path:
        mask_path = mask_path.replace('/mnt/odyssey', '/n/coxfs01/2p-data')
    excluded_tiffs = TID['PARAMS']['excluded_tiffs']

    maskinfo = dict()
    try:
        maskfile = h5py.File(mask_path, 'r')
        is_3D = maskfile.attrs['is_3D'] in ['True']

        # Identify tiff source for ROIs:
        roidict_path = os.path.join(rootdir, maskfile.attrs['animal'], maskfile.attrs['session'], 'ROIs', 'rids_%s.json' % maskfile.attrs['session'])
        with open(roidict_path, 'r') as f:
            roidict = json.load(f)
        roi_tiff_src = roidict[maskfile.attrs['roi_id']]['SRC']
        if rootdir not in roi_tiff_src:
            roi_tiff_src = hutils.replace_root(roi_tiff_src, rootdir, maskfile.attrs['animal'], maskfile.attrs['session'])

        # Check whether ROI tiffs are same src as TRACE ID tiffs:
        trace_tiff_src = TID['SRC']
        if rootdir not in trace_tiff_src:
            trace_tiff_src = hutils.replace_root(trace_tiff_src, rootdir, maskfile.attrs['animal'], maskfile.attrs['session'])

        # Get n tiffs from TRACE source:
        ntiffs = len([f for f in os.listdir(trace_tiff_src) if f.endswith('tif')])

        # Get files from which ROIs were extracted in this set:
        maskfiles = maskfile.keys()
        print("MASK FILES:", len(maskfiles))
        if len(maskfiles) == 1:
            ref_file = maskfiles[0]
            single_reference = True
        else:
            ref_file = None
            single_reference = False

        # Get zproj source base dir:
        # For now, assuming preprocessing + motion-correction output of fmt:
        # <...>_ZPROJ_deinterleaved/Channel01/File003 -- only take up to the Channel-dir
        if '_warped' in RID['PARAMS']['options']['zproj_type']:  
            mask_source_dir = os.path.join('%s_mean_deinterleaved' % RID['PARAMS']['tiff_sourcedir'], 'Channel%02d' % RID['PARAMS']['options']['ref_channel'], 'File%03d' % RID['PARAMS']['options']['ref_file'])
        else:
            if 'source' not in maskfile[maskfile.keys()[0]].attrs.keys():
                mask_source_dir = maskfile[maskfile.keys()[0]]['masks'].attrs['source']
            else:
                mask_source_dir = maskfile[maskfile.keys()[0]].attrs['source']
        if rootdir not in mask_source_dir:
            mask_source_dir = hutils.replace_root(mask_source_dir, rootdir, maskfile.attrs['animal'], maskfile.attrs['session'])
        rid_zproj_basedir = os.path.split(mask_source_dir)[0]

        sigchannel_dirname = os.path.split(rid_zproj_basedir)[-1]

        # Get reference file in current trace id set (just use reference from processed dir)
        if roi_tiff_src == trace_tiff_src:
            print("Extracting traces from ROI source")
            matched_sources = True
            if len(maskfiles) == 1:
                ref_file = maskfiles[0]  # REF FILE just is the one used to extract ROIs
            else:
                ref_file = None          # REF FILE doesn't exist, since ROIs extracted from each tif in set
            zproj_source_dir = rid_zproj_basedir
        else:
            print("Extracting traces from ALT run roi src")
            matched_sources = False
            # Identify which file was used as reference, assuming tiffs were preprocessed and motion-corrected:
            if 'mcorrected' in trace_tiff_src:
                # Walk backward from standard motion-correction output-dir formatting
                # to get filepath parts we need:
                processed_dir = os.path.split(trace_tiff_src.split('/mcorrected')[0])[0]
                process_name =  os.path.split(trace_tiff_src.split('/mcorrected')[0])[1]
                run_name = os.path.split(os.path.split(processed_dir)[0])[-1]
                with open(os.path.join(processed_dir, 'pids_%s.json' % run_name), 'r') as f:
                    pdict = json.load(f)
                ref_file = 'File%03d' % int(pdict[process_name.split('_')[0]]['PARAMS']['motion']['ref_file'])
            # Get corresponding zproj source dir:
            zproj_source_dir = '%s_mean_deinterleaved/%s' % (trace_tiff_src, sigchannel_dirname)

        # Get list of files in current trace set:
        filenames = sorted(['File%03d' % int(i+1) for i in range(ntiffs)], key=hutils.natural_keys)
        filenames = sorted([ f for f in filenames if f not in excluded_tiffs], key=hutils.natural_keys)
        print("Using reference file %s on %i total tiffs." % (ref_file, len(filenames)))
        # Check if masks are split up by slices: (Matlab, manual2D methods are diff)
        if type(maskfile[maskfiles[0]]['masks']) == h5py.Dataset:
            slice_masks = False
        else:
            slice_keys = [s for s in maskfile[maskfiles[0]]['masks'].keys() if 'Slice' in s]
            if len(slice_keys) > 0:
                slice_masks = True
            else:
                slice_masks = False

        # Get slices for which there are ROIs in this set:
        if slice_masks:
            roi_slices = sorted([str(s) for s in maskfile[maskfiles[0]]['masks'].keys()], \
                                    key=hutils.natural_keys)
            print("Found %i slices for this roi set (%s)" % (len(roi_slices), str(roi_slices)))

        else:
            roi_slices = sorted(["Slice%02d" % int(s+1) for s in range(nslices)], \
                                    key=hutils.natural_keys)
    except Exception as e:
        traceback.print_exc()
        print("Error loading mask info...")
        print("Mask path was: %s" % mask_path)
    #finally:
        #maskfile.close()

    maskinfo['filenames'] = filenames
    maskinfo['ref_file'] = ref_file
    maskinfo['is_single_reference'] = single_reference
    maskinfo['is_3D'] = is_3D
    maskinfo['is_slice_format'] = slice_masks
    maskinfo['roi_slices'] = roi_slices
    maskinfo['filepath'] = mask_path
    maskinfo['matched_sources'] = matched_sources
    maskinfo['zproj_source'] = zproj_source_dir
    maskinfo['roi_source_dir'] = mask_source_dir

    return maskinfo

# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------

def load_roi_assignments(animalid, session, fov, retinorun='retino_run1', 
                            rootdir='/n/coxfs01/2p-data'):
   
    roi_assignments=None
    results_fpath = os.path.join(rootdir, animalid, session, fov, retinorun, 
                              'retino_analysis', 'segmentation', 
                              'roi_assignments.json')
    
    assert os.path.exists(results_fpath), \
            "Assignment results not found: %s" % results_fpath
    with open(results_fpath, 'r') as f:
        roi_assignments = json.load(f)
   
    return roi_assignments #, roi_masks_labeled



def get_masks_and_centroids(dk, experiment, traceid='traces001',
                        xlabel='ml_pos', ylabel='ap_pos',
                        rootdir='/n/coxfs01/2p-data'):
    '''
    Load zprojected image, masks (nrois, d2, d2), and centroids for dataset.
    '''
    session, animalid, fovnum = hutils.split_datakey_str(dk)
    fov = 'FOV%i_zoom2p0x' % fovnum

    # Load zimg
    roiid = get_roiid_from_traceid(animalid, session, 'FOV%i_*' % fovnum, 
                                          experiment, traceid=traceid)
    zimg_path = glob.glob(os.path.join(rootdir, animalid, session, \
                                       'ROIs', '%s*' % roiid, 'figures', '*.tif'))[0]
    zimg = tf.imread(zimg_path)
    zimg = zimg[:, :, 1]
    # Load masks for centroids
    masks, _ = load_roi_masks(animalid, session, fov, rois=roiid, 
                                       rois_first=True)
    # Get centroids, better for plotting
    centroids =  calculate_roi_centroids(masks, xlabel=xlabel, ylabel=ylabel)

    return zimg, masks, centroids

def calculate_roi_centroids(masks, xlabel='x', ylabel='y'):
    '''
    Calculate center of soma, then return centroid coords.
    Assumes shape:  nrois, d1, d2 
    '''
    if np.isnan(masks.max()):
        masks[npisnan(masks)] = 0
    centroids=[]
    for roi in range(masks.shape[0]):
        img = masks[roi, :, :].copy()
        x, y = np.where(img>0)
        centroid = ( round(sum(x) / len(x)), round(sum(y) / len(x)) )
        centroids.append(centroid)
    
    nrois_total = masks.shape[0]
    ctr_df = pd.DataFrame(centroids, columns=[ylabel, xlabel], index=range(nrois_total))

    return ctr_df


def load_roi_masks(animalid, session, fov, rois=None, 
                rois_first=True, rootdir='/n/coxfs01/2p-data'):
    '''
    Loads ROI masks (orig) hdf5 file.
    Returns masks, zimg
    '''
    masks=None; zimg=None;
    mask_fpath = glob.glob(os.path.join(rootdir, animalid, session, 
                                'ROIs', '%s*' % rois, 'masks.hdf5'))[0]
    try:
        mfile = h5py.File(mask_fpath, 'r')

        # Load and reshape masks
        reffile = list(mfile.keys())[0]
        masks = mfile[reffile]['masks']['Slice01'][:].T
        #print(masks.shape)

        zimg = mfile[reffile]['zproj_img']['Slice01'][:].T
       
        if rois_first:
            # npix_y, npix_x, nrois_total = masks.shape
            masks_r0 = np.swapaxes(masks, 0, 2)
            masks = np.swapaxes(masks_r0, 1, 2)
    except Exception as e:
        traceback.print_exc()
    finally:
        mfile.close()
 
    return masks, zimg


def load_roi_positions(datakey, roiid=None, traceid='traces001',
                    rootdir='/n/coxfs01/2p-data'):
    '''
    Load pd.DataFrame() containing ALL position info for cells,
    including converted (ml_pos, ap_pos. Loads 'roi_positions' 
    from saved fovinfo file.
    Called: load_roi_centroids()
    '''
    posdf = None
    session, animalid, fovn = hutils.split_datakey_str(datakey)
    if roiid is None:
        roiid = get_roiid_from_traceid(animalid, session, 
                            'FOV%i_zoom2p0x' % fovn, traceid=traceid)
    # create outpath
    roidir = glob.glob(os.path.join(rootdir, animalid, session,
                        'ROIs', '%s*' % roiid))[0]
    fovinfo_fpath = os.path.join(roidir, 'fov_info.pkl')
    try:
        # print("... loading roi coords")
        with open(fovinfo_fpath, 'rb') as f:
            fovinfo = pkl.load(f) #, encoding='latin1')
        assert 'roi_positions' in fovinfo.keys(), "Bad file: %s" % fovinfo_fpath
        posdf = fovinfo['roi_positions'].copy()

    except Exception as e: #AssertionError:
        print("Error loading <%s> for %s" % (roiid, datakey))
        traceback.print_exc()

    return posdf


def get_roi_coords(animalid, session, fov, roiid=None,
                    convert_um=True, traceid='traces001',
                    create_new=False,rootdir='/n/coxfs01/2p-data'):
    '''Prev. called load_roi_coords().
    Loads fovinfo (dict), 'roi_positions' is pd.DataFrame() with all the diff coords'''

    fovinfo = None
    roiid = get_roiid_from_traceid(animalid, session, fov, traceid=traceid)
    # create outpath
    roidir = glob.glob(os.path.join(rootdir, animalid, session,
                        'ROIs', '%s*' % roiid))[0]
    fovinfo_fpath = os.path.join(roidir, 'fov_info.pkl')
    if not create_new:
        try:
            # print("... loading roi coords")
            with open(fovinfo_fpath, 'rb') as f:
                fovinfo = pkl.load(f, encoding='latin1')
            assert 'roi_positions' in fovinfo.keys(), "Bad fovinfo file, redoing"
        except Exception as e: #AssertionError:
            print("Error loading <%s> for %s, %s, %s" % (roiid, animalid, session, fov))
            traceback.print_exc()
            create_new = True

    if create_new:
        print("... calculating roi-2-fov info")
        masks, zimg = load_roi_masks(animalid, session, fov, rois=roiid)
        fovinfo = calculate_roi_coords(masks, zimg, convert_um=convert_um)
        with open(fovinfo_fpath, 'wb') as f:
            pkl.dump(fovinfo, f, protocol=pkl.HIGHEST_PROTOCOL)

    return fovinfo

def get_roiid_from_traceid(animalid, session, fov, run_type=None,
                            traceid='traces001', rootdir='/n/coxfs01/2p-data'):

    if run_type is not None:
        if int(session) < 20190511 and 'rfs' in run_type:
            search_run = 'gratings'
        else:
            search_run = run_type 
        a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session,
                                    fov, '*%s_*' % search_run, 'traces',
                                    'traceids*.json'))[0]
    else:
        a_traceid_dict = glob.glob(os.path.join(rootdir, animalid, session,
                                    fov, '*run*', 'traces', 'traceids*.json'))[0]
    with open(a_traceid_dict, 'r') as f:
        tracedict = json.load(f)

    tid = tracedict[traceid]
    roiid = tid['PARAMS']['roi_id']

    return roiid


def calculate_roi_coords(masks, zimg, roi_list=None, convert_um=True, rois_first=True):
    '''
    Get FOV info relating cortical position to RF position of all cells.
    Info should be saved in: rfdir/fov_info.pkl
    
    Returns:
        fovinfo (dict)
            'roi_positions': dataframe
                fov_xpos: micron-converted fov position (IRL is AP-axis)
                fov_ypos: " " (IRL is ML-axis)
                fov_xpos_pix: coords in pixel space
                fov_ypos_pix': " "
                ml_pos: transformed, rotated coords
                ap_pos: transformed, rotated coors (natural view)
            'zimg': 
                (array) z-projected image 
            'roi_contours': 
                (list) roi contours, classifications.convert_coords.contours_from_masks()
            'xlim' and 'ylim': 
                (float) FOV limits (in pixels or um) for (natural) azimuth and elevation axes
    '''

    if np.isnan(masks.max()):
        masks[npisnan(masks)] = 0
 
    print("... getting fov info")
    # Get masks
    if rois_first:
        nrois_total, npix_y, npix_x = masks.shape
    else:
        npix_y, npix_x, nrois_total = masks.shape

    if roi_list is None:
        roi_list = range(nrois_total)

    # Create contours from maskL
    roi_contours = contours_from_masks(masks, rois_first=rois_first)

    # Convert to brain coords (scale to microns)
    fov_pos_x, fov_pos_y, xlim, ylim, centroids = get_roi_position_in_fov(roi_contours,
                                                               roi_list=roi_list,
                                                                 convert_um=convert_um,
                                                                 npix_y=npix_y,
                                                                 npix_x=npix_x)

    #posdf = pd.DataFrame({'ml_pos': fov_pos_y, 'ap_pos': fov_pos_x, #fov_x,
    posdf = pd.DataFrame({'fov_xpos': fov_pos_x, # corresponds to AP axis ('ap_pos')
                          'fov_ypos': fov_pos_y, # corresponds to ML axis ('ml_pos')
                          'fov_xpos_pix': [c[0] for c in centroids],
                          'fov_ypos_pix': [c[1] for c in centroids]
                          }, index=roi_list)

    posdf = transform_fov_posdf(posdf, ml_lim=ylim, ap_lim=xlim)
    # Save fov info
    pixel_size = hutils.get_pixel_size()
    fovinfo = {'zimg': zimg,
                'convert_um': convert_um,
                'pixel_size': pixel_size,
                'roi_contours': roi_contours,
                'roi_positions': posdf,
                'ap_lim': xlim, # natural orientation AP (since 2p fov is rotated 90d)
                'ml_lim': ylim} # natural orientation ML

    return fovinfo


def contours_from_masks(masks, rois_first=True):
    # Cycle through all ROIs and get their edges
    # (note:  tried doing this on sum of all ROIs, but fails if ROIs are overlapping at all)
    tmp_roi_contours = []
    nrois = masks.shape[0] if rois_first else masks.shape[-1]
    for ridx in range(nrois):
        im = masks[ridx, :, :] if rois_first else masks[:,:,ridx] 
        im[im>0] = 1
        im[im==0] = np.nan #1
        im = im.astype('uint8')
        edged = cv2.Canny(im, 0, 0.9)
        tmp_cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        #tmp_cnts = tmp_cnts[0] if imutils.is_cv2() else tmp_cnts[1]
        tmp_cnts = tmp_cnts[1] if imutils.is_cv2() else tmp_cnts[0]
        tmp_roi_contours.append((ridx, tmp_cnts[0]))
    print("Created %i contours for rois." % len(tmp_roi_contours))

    return tmp_roi_contours


def get_roi_position_in_fov(tmp_roi_contours, roi_list=None,
                            convert_um=True, npix_y=512, npix_x=512):
                            #xaxis_conversion=2.3, yaxis_conversion=1.9):

    '''
    From 20190605 PSF measurement:
        xaxis_conversion = 2.312
        yaxis_conversion = 1.904
    '''
    print("... (not sorting)")

    if not convert_um:
        xaxis_conversion = 1.
        yaxis_conversion = 1.
    else:
        (xaxis_conversion, yaxis_conversion) = hutils.get_pixel_size()

    # Get ROI centroids:
    #print(tmp_roi_contours[0])
    centroids = [get_contour_center(cnt[1]) for cnt in tmp_roi_contours]

    # Convert pixels to um:
    xlinspace = np.linspace(0, npix_x*xaxis_conversion, num=npix_x)
    ylinspace = np.linspace(0, npix_y*yaxis_conversion, num=npix_y)

    xlim=xlinspace.max() if convert_um else npix_x
    ylim = ylinspace.max() if convert_um else npix_y

    if roi_list is None:
        roi_list = [cnt[1] for cnt in tmp_roi_contours] #range(len(tmp_roi_contours)) #sorted_roi_indices_xaxis))
        #print(roi_list[0:10])

    fov_pos_x = [xlinspace[pos[0]] for pos in centroids]
    fov_pos_y = [ylinspace[pos[1]] for pos in centroids]
    
    return fov_pos_x, fov_pos_y, xlim, ylim, centroids

def get_contour_center(cnt):
    cnt =(cnt).astype(np.float32)
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

# ############################################
# Functions for processing ROIs (masks)
# ############################################
def transform_rotate_coordinates(positions, ap_lim=1177., ml_lim=972.):
    # Rotate 90 degrees (ie., rotate counter clockwise about origin: (-y, x))
    # For image, where 0 is at top (y-axis points downward), 
    # then rotate CLOCKWISE, i.e., (y, -x)
    # Flip L/R, too.
    # (y, -x):  (pos[1], 512-pos[0]) --> 512 to make it non-neg, and align to image
    # flip l/r: (512-pos[1], ...) --> flips x-axis l/r 
    positions_t = [(ml_lim-pos[1], ap_lim-pos[0]) for pos in positions]

    return positions_t

def transform_fov_posdf(posdf, fov_keys=('fov_xpos', 'fov_ypos'),
                         ml_lim=972, ap_lim=1177.):
    posdf_transf = posdf.copy()

    fov_xkey, fov_ykey = fov_keys
    fov_xpos = posdf_transf[fov_xkey].values
    fov_ypos = posdf_transf[fov_ykey].values

    o_coords = [(xv, yv) for xv, yv in zip(fov_xpos, fov_ypos)]
    t_coords = transform_rotate_coordinates(o_coords, ap_lim=ap_lim, ml_lim=ml_lim)
    posdf['ml_pos'] = [t[0] for t in t_coords]
    posdf['ap_pos'] = [t[1] for t in t_coords]

    return posdf



