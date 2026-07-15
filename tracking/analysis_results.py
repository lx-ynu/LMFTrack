import argparse

import _init_paths
from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved LMFTrack result files.')
    parser.add_argument('--tracker_name', default='lmftrack')
    parser.add_argument('--tracker_param', default='lmftrack_lasher')
    parser.add_argument('--dataset_name', default='lasher_test')
    parser.add_argument('--runid', type=int, default=15)
    parser.add_argument('--display_name', default='LMFTrack')
    args = parser.parse_args()

    trackers = trackerlist(
        name=args.tracker_name,
        parameter_name=args.tracker_param,
        dataset_name=args.dataset_name,
        run_ids=args.runid,
        display_name=args.display_name,
    )
    dataset = get_dataset(args.dataset_name)
    print_results(
        trackers,
        dataset,
        args.dataset_name,
        merge_results=True,
        plot_types=('success', 'norm_prec', 'prec'),
    )


if __name__ == '__main__':
    main()
