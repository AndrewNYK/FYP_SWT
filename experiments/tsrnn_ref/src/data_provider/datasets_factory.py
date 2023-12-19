"""Data Provider."""

from src.data_provider import aep

datasets_map = {
    'aep': aep,
}

def data_provider(dataset_name,
                  train_data_path,
                  valid_data_path,
                  batch_size,
                  seq_length,
                  img_width,
                  img_height,
                  horizon,
                  is_normalized=True,
                  iweek=0,
                  max_value=None,
                  is_training=True):
  """Returns a Dataset.

  Args:
    dataset_name: String, the name of the dataset.
    train_data_path: String, the path where training data are saved.
    valid_data_path: String, the path where validation data are saved.
    batch_size: Int, the batch size.
    seq_length: Int, the length of the input sequence.
    horizon: Int, the number of future steps to be predicted.
    is_normalized: Bool, normalized by min and max of training instances.
    is_training: Bool, training or testing.

  Returns:
      if is_training is True, it returns two dataset instances for both
      training and evaluation. Otherwise only one dataset instance for
      evaluation.
  Raises:
      ValueError: When `dataset_name` is unknown.
  """
  if dataset_name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % dataset_name)

  if dataset_name == 'aep':
    # load validation data
    test_input_param = {
        'path': valid_data_path,
        'batch_size': 1,#batch_size,
        'seq_length': seq_length,
        'img_width': img_width,
        'img_height': img_height,
        'horizon': horizon,
        'is_normalized': is_normalized,
        'iweek': iweek,
        'max_value': max_value,
        'is_test': True,
        'name': dataset_name + 'test iterator'
    }
    # call input handle function from 'dataset_name' class
    test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
    test_input_handle.begin(do_shuffle=False)

    if is_training:
      # load training data
      train_input_param = {
          'path': train_data_path,
          'batch_size': batch_size,
          'seq_length': seq_length,
          'img_width': img_width,
          'img_height': img_height,
          'horizon': horizon,
          'is_normalized': is_normalized,
          'iweek': iweek,
          'max_value': max_value,
          'is_test': False,
          'name': dataset_name + ' train iterator'
      }
      train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
      train_input_handle.begin(do_shuffle=True)
      return train_input_handle, test_input_handle
    else:
      return test_input_handle