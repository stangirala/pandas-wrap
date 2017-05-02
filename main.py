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

    # TODO handle index setting
    return WrapDataFrame(pd.DataFrame(mapped_rows))

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

def test_select():
  data = {'A' : ['one', 'one', 'two', 'three'],
          'B' : ['A', 'B', 'C', 'D'],
          'C' : ['foo', 'foo', 'bar', 'bar'],
          'D' : [1, 2, 3, 4],
          'E' : np.random.randn(4)}
  df = pd.DataFrame(data)
  wdf = WrapDataFrame(df)

  _wdf = wdf.select('A', 'D')
  expected_pairs = list(zip(data['A'], data['D']))
  for actual_pair, expected_pair in zip(_wdf.values, expected_pairs):
    assert tuple(actual_pair) == expected_pair

  with pytest.raises(Exception):
    _df = wdf.select('some column', 'C')

def test_select_by_position():
  data = {'A' : ['one', 'one', 'two', 'three'],
          'B' : ['A', 'B', 'C', 'D'],
          'C' : ['foo', 'foo', 'bar', 'bar'],
          'D' : [1, 2, 3, 4],
          'E' : np.random.randn(4)}
  df = pd.DataFrame(data)
  wdf = WrapDataFrame(df)

  _wdf = wdf.selectByPosition(1, 2, 3)
  expected_pairs = list(zip(data['B'], data['C'], data['D']))
  for actual_pair, expected_pair in zip(_wdf.values, expected_pairs):
    assert tuple(actual_pair) == expected_pair

  with pytest.raises(Exception):
    _df = wdf.selectByPosition(-1)
    _df = wdf.selectByPosition(5)

def test_map():
  # map functions expect a positional tuple
  data = {'A' : ['one', 'one', 'two', 'three'],
          'B' : ['A', 'B', 'C', 'D'],
          'C' : ['foo', 'foo', 'bar', 'bar'],
          'D' : [1, 2, 3, 4],
          'E' : np.random.randn(4)}
  df = pd.DataFrame(data)
  wdf = WrapDataFrame(df)
  wdf = wdf.map(lambda tup: tup._1+tup._2, ('A', 'C'))
  expected_pairs = list(zip(data['A'], data['C']))
  for actual_str, expected_pair in zip(wdf.values, expected_pairs):
    assert actual_str[0] == ''.join(expected_pair[0]+expected_pair[1])

def test_filter():
  # filter functions expect a positional tuple
  data = {'A' : ['one', 'one', 'two', 'three'],
          'B' : ['A', 'B', 'C', 'D'],
          'C' : ['foo', 'foo', 'bar', 'bar'],
          'D' : [1, 2, 3, 4],
          'E' : np.random.randn(4)}

  df = pd.DataFrame(data)
  wdf = WrapDataFrame(df).select('A', 'B', 'C', 'D')
  wdf = wdf.filter(lambda tup: tup._1 != 'one' and tup._4 != 4)
  expected = ['two', 'C', 'bar', 3]
  assert list(wdf.values[0]) == expected

def test_fold_left():
  # fold functions expect an accumulator and an associative combiner
  data = {'A' : ['one', 'one', 'two', 'three'],
          'B' : ['A', 'B', 'C', 'D'],
          'C' : ['foo', 'foo', 'bar', 'bar'],
          'D' : [1, 2, 3, 4],
          'E' : np.random.randn(4)}

  df = pd.DataFrame(data)
  wdf = WrapDataFrame(df).select('D')
  foldValue = wdf.foldLeft(0, lambda acc, tup: acc+tup._1)
  assert foldValue == 10
