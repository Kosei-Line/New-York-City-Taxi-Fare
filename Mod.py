import argparse

parser = argparse.ArgumentParser(description='Titanic disastar')
parser.add_argument('--gpu', '-g', default=0, type=int)
parser.add_argument('--out', '-o', default='result', type=str)
parser.add_argument('--epoch', '-e', default=100, type=int)
parser.add_argument('--batch', '-b', default=256, type=int)
parser.add_argument('--snapshot', '-s', default=1, type=int)

args = parser.parse_args()

print('#GPU:{}'.format(args.gpu))
print('#batchsize:{}'.format(args.batch))
print('#epoch:{}'.format(args.epoch))
print('')

import Net as Net
import Updater as Updater
import Evaluator as Evaluator