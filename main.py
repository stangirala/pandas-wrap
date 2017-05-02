import pandas as pd
import numpy as np

class WrapDataFrame():
  def __init__(self, df):
    self.dataframe = df
    self.dataframe_columns = df.columns

  def select(self, *args):
    for arg in args:
      if arg not in self.dataframe_columns:
        raise Exception('Column not found: ' + arg)

    return self.dataframe[[i for i in args]]

  def map(self, map_function, select_columns):
    for column in select_columns:
      if column not in self.dataframe_columns:
        raise Exception('Could not select column: ' + column)

    df = self.select(*select_columns)
    value_rows = df.values
    mapped_rows = []

    for value_row in value_rows:
      # TODO add map failure
      mapped_row = map_function(value_row)
      mapped_rows.append(mapped_row)

    # TODO handle index setting
    return pd.DataFrame(mapped_rows)

def test_select(wdf):
  _df = wdf.select('A', 'C')
  try:
    _df = wdf.select('some column', 'C')
  except Exception:
    pass

if __name__ == '__main__':
  df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                            'B' : ['A', 'B', 'C'] * 4,
                            'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                            'D' : np.random.randn(12),
                            'E' : np.random.randn(12)})

  wdf = WrapDataFrame(df)
  test_select(wdf)

  print(wdf.map(lambda x: x+x, ('A', 'C')))
