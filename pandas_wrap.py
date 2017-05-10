import pandas as pd
import numpy as np
import pytest

class PositionTuple:
  def __init__(self, *args):
    for pos, arg in enumerate(args):
      setattr(self, '_'+str(pos+1), arg)

  def __str__(self):
    return ' '.join(','.join(map(str, [k, v])) for k, v in self.__dict__.items())

class WrapDataFrame():
  def __init__(self, df):
    self.dataframe = df
    self.dataframe_columns = df.columns
    self.values = self.dataframe.values
    self.size = len(self.dataframe)

  def select(self, *args):
    for arg in args:
      if arg not in self.dataframe_columns:
        raise Exception('Column not found: ' + arg)

    return WrapDataFrame(self.dataframe[list(args)])

  def selectByPosition(self, *args):
    if not all([str(arg).isdigit() for arg in args]):
      raise Exception('Column position was not a number')

    positions = sorted(args)
    if len(self.dataframe_columns) <= args[-1] or args[-1] < 0:
      raise Exception('Max column position > number of available columns')

    return WrapDataFrame(self.dataframe[list(positions)])

  def map(self, map_function, select_columns):
    for column in select_columns:
      if column not in self.dataframe_columns:
        raise Exception('Could not select column: ' + column)

    wdf = self.select(*select_columns)
    value_rows = wdf.values
    mapped_rows = []

    for value_row in value_rows:
      mapped_row = map_function(PositionTuple(*value_row.tolist()))
      mapped_rows.append(mapped_row)

    # TODO fix index setting
    return WrapDataFrame(pd.DataFrame(mapped_rows))

  def typed_map(self, map_function, return_type, select_columns):
    for column in select_columns:
      if column not in self.dataframe_columns:
        raise Exception('Could not select column: ' + column)

    # Assume return_type is a numpy dtype and uses that to handle allocated numpy arrays
    assert type(return_type) == np.dtype
    mapped_data_array = np.empty(shape=(self.size,), dtype=return_type)

    wdf = self.select(*select_columns)
    value_rows = wdf.values
    for ind, value_row in enumerate(value_rows):
      mapped_row = map_function(PositionTuple(*value_row.tolist()))
      mapped_data_array[ind] = mapped_row

    # TODO fix index setting
    return WrapDataFrame(pd.DataFrame(mapped_data_array))

  def filter(self, filter_function):
    value_rows = self.dataframe.values
    filtered_row = []

    for value_row in value_rows:
      pos_tuple = PositionTuple(*value_row.tolist())
      if filter_function(pos_tuple):
        filtered_row.append(value_row)

    return WrapDataFrame(pd.DataFrame(filtered_row))

  def foldLeft(self, acc_zero_value, fold_function):
    value_rows = self.dataframe.values

    acc = acc_zero_value

    for value_row in value_rows:
      pos_tuple = PositionTuple(*value_row.tolist())
      acc = fold_function(acc, pos_tuple)

    return acc

  def __str__(self):
    return str(self.dataframe)
