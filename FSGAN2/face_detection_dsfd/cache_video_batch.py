import os
from glob import glob
import traceback
import logging
import cache_video


def main(in_dir, out_dir=None, indices=None, detection_model_path='weights/WIDERFace_DSFD_RES152.pth', batch_size=8,
         out_postfix='_dsfd.pkl'):
    out_dir = in_dir if out_dir is None else out_dir
    vid_paths = sorted(glob(os.path.join(in_dir, '*.mp4')))
    vid_paths = eval('vid_paths[%s]' % indices) if indices is not None else vid_paths

    # For each video file
    for i, vid_path in enumerate(vid_paths):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        curr_cache_path = os.path.join(out_dir, vid_name + out_postfix)

        if os.path.exists(curr_cache_path):
            print('[%d/%d] Skipping "%s"' % (i + 1, len(vid_paths), vid_name))
            continue
        else:
            print('[%d/%d] Processing "%s"...' % (i + 1, len(vid_paths), vid_name))

        # Process video
        try:
            cache_video.main(vid_path, curr_cache_path, detection_model_path, batch_size)
        except Exception as e:
            logging.error(traceback.format_exc())


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('cache_video_batch')
    parser.add_argument('input', metavar='DIR',
                        help='path input directory')
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output directory')
    parser.add_argument('-i', '--indices', default=None,
                        help='python style indices (e.g 0:10')
    parser.add_argument('-dm', '--detection_model', metavar='PATH', default='weights/WIDERFace_DSFD_RES152.pth',
                        help='path to face detection model')
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                        help='batch size (default: 8)')
    parser.add_argument('-op', '--out_postfix', default='_dsfd.pkl', metavar='POSTFIX',
                        help='output file postfix')
    args = parser.parse_args()
    main(args.input, args.output, args.indices, args.detection_model, args.batch_size, args.out_postfix)
