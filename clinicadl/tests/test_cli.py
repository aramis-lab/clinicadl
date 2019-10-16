import clinicadl.cli as cli

def test_cli_preprocessing():
  parser = cli.parse_command_line()
  args = parser.parse_args(['preprocessing',
      '/dir/bids',
      '/dir/caps', 
      '/dir/tsv.file', 
      '/dir/ref.template',
      '/dir/work/dir'])
  arguments = vars(args)
  assert (arguments["bids_directory"]=="/dir/bids" and 
          arguments["caps_dir"]=="/dir/caps" and
          arguments["tsv_file"]=="/dir/tsv.file" and
          arguments["ref_template"]=="/dir/ref.template" and
          arguments["working_directory"]=="/dir/work/dir")

def test_cli_extract():
  parser = cli.parse_command_line()
  patch_size = 50
  stride_size = 50
  extract_method = 'slice'
  slice_direction = 0
  slice_mode = 'rgb'
  args = parser.parse_args(['extract',
      '/dir/caps', 
      '/dir/tsv.file',
      '/dir/work/dir',
      extract_method,
      '-psz', '50',
      '-ssz', '50',
      '-sd', '0',
      '-sm', slice_mode])
  arguments = vars(args)
  assert (arguments["caps_dir"]=="/dir/caps" and
          arguments["tsv_file"]=="/dir/tsv.file" and
          arguments["patch_size"]==patch_size and
          arguments["stride_size"]==stride_size and
          arguments["working_directory"]=="/dir/work/dir" and
          arguments["extract_method"]==extract_method and
          arguments["slice_direction"]==slice_direction and
          arguments["slice_mode"]==slice_mode
          )
 
  
def test_cli_train_subject():
  parser = cli.parse_command_line()
  pass

def test_cli_train_slice():
  parser = cli.parse_command_line()
  pass

def test_cli_train_patch():
  parser = cli.parse_command_line()
  pass

def test_cli_train_svn():
  parser = cli.parse_command_line()
  pass
