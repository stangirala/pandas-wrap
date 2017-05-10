import pandas as pd
import numpy as np
import pytest
import random

from pandas_wrap import WrapDataFrame

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
    wdf.select('some column', 'C')

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
    wdf.selectByPosition(-1)
    wdf.selectByPosition(5)

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

def test_typed_map():
  # map functions expect a positional tuple
  data = {'A' : ['one', 'one', 'two', 'three'],
          'B' : ['A', 'B', 'C', 'D'],
          'C' : ['foo', 'foo', 'bar', 'bar'],
          'D' : [1, 2, 3, 4],
          'E' : np.random.randn(4)}
  df = pd.DataFrame(data)
  wdf = WrapDataFrame(df)

  map_return_type = np.dtype('a20')

  wdf = wdf.typed_map(lambda tup: tup._1+tup._2, map_return_type, ('A', 'C'))
  expected_pairs = list(zip(data['A'], data['C']))
  for actual_str, expected_pair in zip(wdf.values, expected_pairs):
    assert actual_str[0].decode('utf-8') == ''.join(expected_pair[0]+expected_pair[1])

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

def map_operation(wdf):
  foldValue = wdf.foldLeft(0, lambda acc, tup: acc+tup._1)

def typed_map_operation(wdf):
  foldValue = wdf.foldLeft(0, lambda acc, tup: acc+tup._1)

def test_map_perf():
    import timeit
    # Early 2015 MBP
    number = 50000
    map_op = timeit.timeit('map_operation(m)', number=number, setup='from __main__ import map_operation, m')
    typed_map = timeit.timeit('typed_map_operation(tm)', number=number, setup='from __main__ import typed_map_operation, tm')
    print('map_op', map_op)
    print('typed_map', typed_map)

if __name__ == '__main__':
    data = {'A' : ['one', 'one', 'two', 'three'],
          'B' : ['A', 'B', 'C', 'D'],
          'C' : ['foo', 'foo', 'bar', 'bar'],
          'D' : [1, 2, 3, 4],
          'E' : np.random.randn(4)}
    df = pd.DataFrame(data)
    wdf = WrapDataFrame(df)
    m = wdf.map(lambda tup: (tup._1+tup._2, tup._1**2), ('D', 'E'))

    map_return_type = np.dtype([('f1', 'u8'), ('f2', 'u8')])
    tm = wdf.typed_map(lambda tup: (tup._1+tup._2, tup._1**2), map_return_type, ('D', 'E'))

    test_map_perf()
