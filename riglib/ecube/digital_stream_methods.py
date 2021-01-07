import numpy as np
import pyeCubeStream

def ts_values(data, srate):
  """Finds the timestamp and corresponding value
  of all the bit flips in data
  author Leo
  """
  logical_idx = np.insert(np.diff(data) != 0, 0, True)
  time = np.arange(np.size(data))/srate
  return time[logical_idx], data[logical_idx]


def ffs(x):
  """Returns the index, counting from 0, of the
  least significant set bit in `x`.
  author leo
  """
  return (x & -x).bit_length() - 1

def mask_and_shift(data, bit_mask):
  """Apply bit mask and shift to the least
  significant set bit
  l
  """
  return np.bitwise_and(data, bit_mask) >> ffs(bit_mask)

if __name__ == "__main__":

    ecubeDigital = pyeCubeStream.eCubeStream('DigitalPanel')

    ecubeDigital.start()
    sample = ecubeDigital.get()
    last_sample = sample[1][-1]
    print(f'last digital value of all channels: {last_sample}')

    long_data = np.squeeze(np.int_(last_sample))
    print(f'after some conversion: {long_data}')

    masks = [0xff, 0xff00, 0xff0000, 0xff000000]

    #code_num_1 = mask_and_shift(long_data,0xff)
    

    print(f'print the channels of in groups of 8')
    print('MSB <- LSB')
    for m in masks:
        code_num_1 = mask_and_shift(long_data,m)
        print(f'{np.binary_repr(code_num_1)}')

    #ts, values = ts_values(long_data, dat.samplerate)

