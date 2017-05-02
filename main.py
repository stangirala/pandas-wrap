import pandas as pd
import numpy as np

class WrapDataFrame():
  def __init__(self, df):
    self.dataframe = df
    self.dataframe_columns = df.columns

  def values(self):
    return self.dataframe.values

  def select(self, *args):
    for arg in args:
      if arg not in self.dataframe_columns:
        raise Exception('Column not found: ' + arg)

    return WrapDataFrame(self.dataframe[list(args)])

  def select_by_position(self, *args):
    if not all([str(arg).isdigit() for arg in args]):
      raise Exception('Column position was not a number')

    positions = sorted(args)
    if len(self.dataframe_columns) <= args[-1]:
      raise Exception('Max column position > number of available columns')

    return WrapDataFrame(self.dataframe[list(positions)])

  def map(self, map_function, select_columns):
    for column in select_columns:
      if column not in self.dataframe_columns:
        raise Exception('Could not select column: ' + column)

    df = self.select(*select_columns)
    value_rows = df.values()
    mapped_rows = []

    for value_row in value_rows:
      # TODO add map failure
      mapped_row = map_function(tuple(value_row))
      mapped_rows.append(mapped_row)

    # TODO handle index setting
    return WrapDataFrame(pd.DataFrame(mapped_rows))

  def __str__(self):
    return str(self.dataframe)

def test_select(wdf):
  _df = wdf.select('A', 'C')
  try:
    _df = wdf.select('some column', 'C')
  except Exception:
    pass

def test_map(wdf):
  print(wdf.map(lambda x: x[0]+x[1], ('A', 'C')))

if __name__ == '__main__':
  df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                            'B' : ['A', 'B', 'C'] * 4,
                            'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                            'D' : np.random.randn(12),
                            'E' : np.random.randn(12)})

  wdf = WrapDataFrame(df)
  #test_select(wdf)
  #test_map(wdf)

  print(wdf.select_by_position(1, 2, 3))
