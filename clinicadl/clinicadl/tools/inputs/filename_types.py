# coding: utf8

FILENAME_TYPE = {'full': '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w',
                 'cropped': '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w',
                 'skull_stripped': '_space-Ixi549Space_desc-skullstripped_T1w',
                 'gm_maps': '_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability',
                 'shepplogan': '_phantom-SheppLogan'}

MASK_PATTERN = {'full': '_res-1x1x1',
                'cropped': '_desc-Crop_res-1x1x1',
                'skull_stripped': '',
                'gm_maps': '',
                'shepplogan': ''}
