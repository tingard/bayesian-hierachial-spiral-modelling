import numpy as np
import argparse
parser = argparse.ArgumentParser(
    description=(
        'Select a random subsample of '
    )
)
parser.add_argument('-n', metavar='N', type=int, required=True,
                    help='path to save traces (if desired)')
args = parser.parse_args()

print('Generating subsample of {} subjects'.format(args.n))
subjects = np.loadtxt('lib/subject-id-list.csv', dtype='u8')
sample = np.random.choice(subjects, args.n)
np.savetxt('./subject-subset.csv', sample, fmt='%d')
