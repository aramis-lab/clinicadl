# coding: utf8

def test_preprocessing_t1w():
    from clinicadl.preprocessing.T1_lineal import preprocessing_t1w

    bids_dir = './test/bids/'
    caps_dir = './test/caps/'
    tsv_file = './test/subjects.tsv'
    working_dir = './test/working_dir'
    nproc = 2

    wf = preprocessing_t1w(args.bids_directory,
            args.caps_dir,
            args.tsv_file,
            args.working_directory)
    wf.run(plugin='MultiProc', plugin_args={'n_procs': nproc})

    pass
