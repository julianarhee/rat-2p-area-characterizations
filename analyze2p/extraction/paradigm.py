#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:31:35 2018

@author: juliana
"""

def format_stimconfigs(configs):
    
    stimconfigs = copy.deepcopy(configs)
    cfg_list = sorted(list(stimconfigs.keys()))
    sample_key = cfg_list[0]
    stim_params = list(stimconfigs[sample_key].keys())
   
    if 'frequency' in stim_params:
        stimtype = 'gratings'
    elif 'fps' in stim_params:
        stimtype = 'movie'
    else:
        stimtype = 'image' 
    print "STIM TYPE:", stimtype 
    # Split position into x,y:
    for cf in cfg_list: #stimconfigs.keys():
        stimconfigs[cf]['xpos'] = None if configs[cf]['position'][0] is None \
                                    else round(configs[cf]['position'][0], 2)
        stimconfigs[cf]['ypos'] = None if configs[cf]['position'][1] is None \
                                    else round(configs[cf]['position'][1], 2) 
        stimconfigs[cf]['size'] = None if configs[cf]['scale'][0] is None \
                                    else configs[cf]['scale'][0]
        if 'aspect' in configs[cf].keys() and stimconfigs[cf]['size'] is not None:
            stimconfigs[cf]['size'] = stimconfigs[cf]['size']/configs[cf]['aspect']
        #stimconfigs[config].pop('position', None)
        stimconfigs[cf].pop('scale', None)
        stimconfigs[cf]['stimtype'] = stimtype
        if 'color' in configs[cf].keys():
            stimconfigs[cf]['luminance'] = round(configs[cf]['color'], 3) \
                                            if isinstance(configs[cf]['color'], float) else None
        else:
            stimconfigs[cf]['luminance'] = None
        
        # stimulus-type specific variables:
        if stimtype == 'gratings':
            stimconfigs[cf]['sf'] = configs[cf]['frequency']
            stimconfigs[cf]['ori'] = configs[cf]['rotation']
            stimconfigs[cf].pop('frequency', None)
            stimconfigs[cf].pop('rotation', None)
        else:
            transform_variables = ['object', 'xpos', 'ypos', 'size', 'yrot', 'morphlevel', 'stimtype', 'color']

            # Figure out Morph IDX for 1st and last anchor image:
            image_names = list(set([configs[c]['filename'] for c in cfg_list]))
            im_exts = [os.path.splitext(configs[k]['filename'])[-1]=='.png' \
                        for k in cfg_list]
            if any(im_exts):
                if len(image_names) >= 2:
                    if any(['Blob_N1' in i for i in image_names]) \
                            and any(['Blob_N2' in i for i in image_names]):
                        mlevels = []
                        anchor1 = 0
                        anchor2 = 106
                else:
                    fns = [configs[c]['filename'] for c in cfgf_list \
                            if 'morph' in configs[c]['filename']]
                    mlevels = sorted(list(set([int(fn.split('_')[0][5:]) \
                            for fn in fns])))
            elif 'fps' in stim_params:
                fns = [configs[c]['filename'] for c in cfg_list \
                            if 'Blob_M' in configs[c]['filename']]
                mlevels = sorted(list(set([int(fn.split('_')[1][1:]) \
                            for fn in fns])))    
            
            if len(mlevels) > 0:
                anchor2 = 106 if mlevels[-1]>22 else 22
            assert all([anchor2>m for m in mlevels]), \
                "Possibly incorrect morphlevel assignment (%i). Found morphs %s." \
                    % (anchor2, str(mlevels))

            if stimtype == 'image':
                imname = os.path.splitext(configs[config]['filename'])[0]
                if ('CamRot' in imname):
                    objectid = imname.split('_CamRot_')[0]
                    yrot = int(imname.split('_CamRot_y')[-1])
                    if 'N1' in imname or 'D1' in imname:
                        morphlevel = 0
                    elif 'N2' in imname or 'D2' in imname:
                        morphlevel = anchor2
                    elif 'morph' in imname:
                        morphlevel = int(imname.split('_CamRot_y')[0]\
                                        .split('morph')[-1])   
                elif '_zRot' in imname:
                    # Real-world objects:  format is 'IDENTIFIER_xRot0_yRot0_zRot0'
                    objectid = imname.split('_')[0]
                    yrot = int(imname.split('_')[3][4:])
                    morphlevel = 0
                elif 'morph' in imname: 
                    # These are morphs w/ old naming convention, 
                    # 'CamRot' not in filename)
                    if '_y' not in imname and '_yrot' not in imname:
                        objectid = imname #'morph' #imname
                        yrot = 0
                        morphlevel = int(imname.split('morph')[-1])
                    else:
                        objectid = imname.split('_y')[0]
                        yrot = int(imname.split('_y')[-1])
                        morphlevel = int(imname.split('_y')[0].split('morph')[-1])
                elif configs[config]['filename']=='' \
                        and configs[config]['stimulus']=='control':
                    objectid = 'control'
                    yrot = 0
                    morphlevel = -1
                     
            elif stimtype == 'movie':
                imname = os.path.splitext(configs[config]['filename'])[0]
                objectid = imname.split('_movie')[0] 
                #'_'.join(imname.split('_')[0:-1])
                if 'reverse' in imname:
                    yrot = -1
                else:
                    yrot = 1
                if imname.split('_')[1] == 'D1':
                    morphlevel = 0
                elif imname.split('_')[1] == 'D2':
                    morphlevel = anchor2
                elif imname.split('_')[1][0] == 'M':
                    # Blob_M11_Rot_y_etc.
                    morphlevel = int(imname.split('_')[1][1:])
                elif imname.split('_')[1] == 'morph':
                    # This is a full morph movie:
                    morphlevel = -1
                    
            stimconfigs[cf]['object'] = objectid
            stimconfigs[cf]['yrot'] = yrot
            stimconfigs[cf]['morphlevel'] = morphlevel
            stimconfigs[cf]['stimtype'] = stimtype
        
            for skey in stimconfigs[cf].keys():
                if skey not in transform_variables:
                    stimconfigs[cf].pop(skey, None)

    return stimconfigs




def get_transforms(stimconfigs):
    
    cfg_list = list(stimconfigs.keys())
    sample_key = cfg_list[0]
    stim_params = list(stimconfigs[sample_key].keys())

    if 'frequency' in stim_params or 'ori' in stim_params:
        stimtype = 'grating'
#    elif 'fps' in stimconfigs[stimconfigs.keys()[0]]:
#        stimtype = 'movie'
    else:
        stimtype = 'image'


    if 'position' in stim_params and ('xpos' not in stim_params or 'ypos' not in stim_params):
        # Need to reformat scnofigs:
        sconfigs = format_stimconfigs(stimconfigs)
    else:
        sconfigs = stimconfigs.copy()
    
    transforms = {'xpos': [x for x in list(set([v['xpos'] \
                                for c, v in sconfigs.items()])) if x is not None],
                       'ypos': [x for x in list(set([v['ypos'] \
                                for c, v in sconfigs.items()])) if x is not None],
                       'size': [x for x in list(set(([v['size'] \
                                for c, v in sconfigs.items()]))) if x is not None]
    }
    
    if stimtype == 'image':
        transforms['yrot'] = np.unique([v['yrot'] for c, v in sconfigs.items()])
        transforms['morphlevel'] = np.unique([v['morphlevel'] \
                                        for c, v in sconfigs.items()])
    else:
        transforms['ori'] = sorted(np.unique([v['ori'] \
                                        for c, v in sconfigs.items()]))
        transforms['sf'] = sorted(np.unique([v['sf'] \
                                        for c, v in sconfigs.items()])))
        if 'stim_dur' in stim_params:
            transforms['direction'] = sorted(np.unique([v['direction'] \
                                        for c, v in sconfigs.items()])))
            transforms['duration'] = sorted(np.unique([v['stim_dur'] \
                                        for c, v in sconfigs.iteritems()]))
        
    trans_types = [t for t in transforms.keys() if len(transforms[t]) > 1]

    object_transformations = {}
    for trans in trans_types:
        if stimtype == 'image':
            curr_objects = []
            for transval in transforms[trans]:
                curr_configs = [c for c,v in sconfigs.iteritems() \
                                        if v[trans]==transval]
                tmp_obj = [np.unique([sconfigs[c]['object'] \
                                        for c in curr_configs])) \
                                        for t in transforms[trans]]
                tmp_obj = list(itertools.chain(*tmp_obj))
                curr_objects.append(tmp_obj)
                
            if len(list(itertools.chain(*curr_objects)))==len(transforms[trans]):
                # There should be a one-to-one correspondence 
                # between object id and the transformation (i.e., morphs)
                included_objects = list(itertools.chain(*curr_objects))
#            elif trans == 'morphlevel':
#                included_objects = list(set(list(itertools.chain(*curr_objects))))
            else:
                included_objects = list(set(curr_objects[0]).intersection(*curr_objects[1:]))
        else:
            included_objects = transforms[trans]
            # print included_objects
        object_transformations[trans] = included_objects

    return transforms, object_transformations



