import os
import glob
import argparse
from collections import defaultdict

from tensorboard.backend.event_processing import event_accumulator
import numpy as np

# example : python util/aggregate_tensorboard_results.py --log-folder $LOG_DIR/baseline --scalar-names val_metric --filter-string *fold_None_*
# --tensorboard-field histograms --scalar-names max_grad_abs --log-folder C:\Temp\hist_grad
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--log-folder', type=str,
                        help="Path to the directory which contains the tensorboard logs to be aggregated.")

    # parser.add_argument('--log-names', type=list, default=None,
    #                     help="The name of the tensorboard logs to aggregate.")

    parser.add_argument('--scalar-names', type=str, nargs='*', default=[],
                        help="The list of scalar names to calculate the mean and standard deviation for.")

    parser.add_argument('--filter-string', type=str, default='*',
                        help='String pattern to pass to glob for automatically finding all relevant tensorboard logs. '
                             'E.g. *_batch2*')

    parser.add_argument('--tensorboard-field', type=str, default="scalars",
                        help='what tab/sheet to extract out : Scalars, histogram')

    args = parser.parse_args()

    # assert not (args.log_names is not None and args.filter_string is not None), \
    #     "Either give --log-names or --filter-string"
    log_paths = glob.glob(os.path.join(args.log_folder, args.filter_string))
    # log_paths = glob.glob(os.path.join(args.log_folder, '*', '*.tfevents.*'))

    print("Using {} logs:".format(len(log_paths)))
    print("\n".join(sorted(log_paths)), "\n\n")

    scalars = None
    scalar_values = defaultdict(list)
    for log_path in log_paths:
        print(log_path)
        event_path = os.path.join(log_path, os.listdir(log_path)[-1])
        ea = event_accumulator.EventAccumulator(log_path,
                                                size_guidance={
                                                    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                    event_accumulator.IMAGES: 4,
                                                    event_accumulator.AUDIO: 4,
                                                    event_accumulator.SCALARS: 0,
                                                    event_accumulator.HISTOGRAMS: 1,
                                                })

        ea.Reload()

        if scalars is None:
            # scalars = ea.Tags()['scalars']
            scalars = ea.Tags()[args.tensorboard_field]
        else:
            pass  #assert set(scalars) == set(ea.Tags()['scalars'])

        if args.tensorboard_field == 'scalars':
            for scalar in scalars:
                scalar_values[scalar].append(ea.Scalars(scalar)[-1].value)

            if len(args.scalar_names) > 0:
                assert len(set(args.scalar_names) - set(scalars)) == 0

                scalars = args.scalar_names
            else:
                scalars = sorted(scalars)

            for scalar_name in scalars:
                np_array = np.array(scalar_values[scalar_name])
                print("{}: Mean: {:.4f}  Std: {:.4f}  Min: {:.4f}  Max: {:.4f}".format(
                    scalar_name, np_array.mean(), np_array.std(), np_array.min(), np_array.max(),
                ))

        elif args.tensorboard_field == 'histograms':
            indices = [i for i, s in enumerate(scalars) if 'grad' in s]

            from operator import itemgetter
            scalars_grad = itemgetter(*np.array(indices).astype('int'))(scalars)
            max_grad = []
            for scalar in scalars_grad:
                scalar_values[scalar].append([ea.Histograms(scalar)[0][2][0], ea.Histograms(scalar)[0][2][1]])
                max_grad.append(np.max(np.abs([ea.Histograms(scalar)[0][2][0], ea.Histograms(scalar)[0][2][1]])))
                # print("TP : {} min={%2.3f} max={}".format(scalar, ea.Histograms(scalar)[0][2][0], ea.Histograms(scalar)[0][2][1]))
                print("TP : {} max|abs|={:.2e} \t min={:.2e} max={:.2e}".format(scalar, np.max(np.abs(
                    [ea.Histograms(scalar)[0][2][0], ea.Histograms(scalar)[0][2][1]])), ea.Histograms(scalar)[0][2][0],ea.Histograms(scalar)[0][2][1]))
            print(np.sort(max_grad)[::-1])

        else:
            raise ValueError()
