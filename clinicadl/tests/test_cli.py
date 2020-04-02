import pytest
import clinicadl.cli as cli
from clinicadl.tools.deep_learning.iotools import Parameters

@pytest.fixture(params=['preprocessing',
    'extract',
    'generate',
    'train_subject',
    'train_slice',
    'train_patch'])
#    'classify'])
def generate_cli_commands(request):
  if request.param == 'preprocessing':
      test_input = [
              'preprocessing',
              '/dir/bids/',
              '/dir/caps/',
              '/dir/file.tsv',
              '/dir/workdir/' ]
      keys_output = [
              'task',
              'bids_dir',
              'caps_dir',
              'tsv_file',
              'working_dir']

  if request.param == 'extract':
      test_input = [
              'extract',
              '/dir/caps', 
              '/dir/tsv.file',
              '/dir/work/dir',
              'slice',
              '-psz', '50',
              '-ssz', '50',
              '-sd', '0',
              '-sm', 'rgb']
      keys_output = [
              'task',
              'caps_dir',
              'tsv_file',
              'working_dir',
              'extract_method',
              'patch_size',
              'stride_size',
              'slice_direction',
              'slice_mode']

  if request.param == 'generate':
      test_input = [
              'generate',
              'random',
              '/dir/caps', 
              '/dir/tsv_path/',
              '/dir/output/',
              '--n_subjects', '10',
              '--preprocessing', '0',
              '--mean', '0.5',
              '--sigma', '0.5']
      keys_output = [
              'task',
              'mode',
              'caps_dir',
              'tsv_path',
              'output_dir',
              'n_subjects',
              'preprocessing',
              'mean',
              'sigma']
  if request.param == 'train_slice':
      test_input = [
              'train',
              'slice',
              '/dir/caps', 
              '/dir/tsv_path/',
              '/dir/output/',
              'Conv5_FC3']
      keys_output = [
              'task',
              'mode',
              'caps_dir',
              'tsv_path',
              'output_dir',
              'network']
  if request.param == 'train_subject':
      test_input = [
              'train',
              'subject',
              '/dir/caps', 
              '/dir/tsv_path/',
              '/dir/output/',
              'Conv5_FC3']
      keys_output = [
              'task',
              'mode',
              'caps_dir',
              'tsv_path',
              'output_dir',
              'network']
  if request.param == 'train_patch':
      test_input = [
              'train',
              'patch',
              '/dir/caps', 
              '/dir/tsv_path/',
              '/dir/output/',
              'Conv5_FC3']
      keys_output = [
              'task',
              'mode',
              'caps_dir',
              'tsv_path',
              'output_dir',
              'network']
  
  return test_input, keys_output


def test_cli(generate_cli_commands):
  import re  
  test_input = generate_cli_commands[0]
  keys_output = generate_cli_commands[1]
  print('Value of test_input is:', type(test_input), test_input)
  regex = re.compile(r'\-.*$')
  test_input_filtered = [i for i in test_input if not regex.match(i)]
  parser = cli.parse_command_line()
  args = parser.parse_args(test_input)
  arguments = vars(args)
  outputs = [str(arguments[x]) for x in keys_output]
  print(outputs)
  assert outputs == test_input_filtered

