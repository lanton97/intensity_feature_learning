"""
    Extract Ouster intensity images from ROS1 (.bag) or ROS2 (dir / .db3 / .mcap) bags into
    8-bit single-channel PNGs for training, optionally tagging each scan with its ground-truth pose.

    Reads the destaggered image topics published by ouster-ros (e.g. GEODE device beta, OS1-64):
        /ouster/reflec_image   (sensor_msgs/Image, mono16)  -> signal we train on
        /ouster/range_image    (sensor_msgs/Image, mono16)  -> no-return mask

    Output (flat folder, what AugmentationPipe globs):
        <out>/<bag>_<stamp_ns>.png        uint8, 0 = never used, 1 = no return, 2..255 = normalized signal
        <out>/masks/<bag>_<stamp_ns>.png  uint8 valid mask (255 = return), with --save_masks
        <out>/index.csv                   file, bag, stamp_ns, gt pose (if --gt), gt_dt_ms
        <out>/<bag>_metadata.json         /ouster/metadata if present (beam angles, lidar->sensor transform)

    --gt takes TUM-format trajectories (t x y z qx qy qz qw; GEODE traj/*.txt). The pose is interpolated
    at the scan header stamp and stored as given: it is the GT body pose, NOT the lidar pose. Apply the
    rig extrinsic (e.g. T_IMU_LiDAR in calibration_files/<device>_config.yaml) when building pairs.

    Examples:
        python extract_lidar_images.py --bags ../geode_bridge/<bag>.bag \
            --gt ../geode_bridge/traj/Bridge/<seq>.txt --out /home/asrl/datasets/lidar/geode_beta --save_masks

        python extract_lidar_images.py --bags ... --list_topics        # inspect a bag first
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation, Slerp


GT_COLS = ['gt_x', 'gt_y', 'gt_z', 'gt_qx', 'gt_qy', 'gt_qz', 'gt_qw', 'gt_dt_ms']


def parse_args():
    p = argparse.ArgumentParser(description='Extract Ouster intensity images from rosbags.')
    p.add_argument('--bags', nargs='+', required=True,
                   help='ROS1 .bag files and/or ROS2 bag directories (or their .db3/.mcap files).')
    p.add_argument('--out', default=None, help='Output folder (required unless --list_topics).')
    p.add_argument('--list_topics', action='store_true', help='Print topics and types, then exit.')

    p.add_argument('--signal_topic', default='/ouster/reflec_image',
                   help='Image topic to train on (reflectivity recommended; e.g. /ouster/signal_image).')
    p.add_argument('--range_topic', default='/ouster/range_image',
                   help="Range image used for the no-return mask. Pass '' to treat signal==0 as no return.")
    p.add_argument('--min_range', type=float, default=0,
                   help='Treat returns closer than this (raw range-image units) as no return, e.g. to drop '
                        'the rig/operator. GEODE beta: 150 masks the carrier, ~0.3%% of scene returns.')
    p.add_argument('--metadata_topic', default='/ouster/metadata',
                   help='Ouster sensor metadata (std_msgs/String JSON: beam angles, lidar->sensor transform). '
                        "Saved as <out>/<bag>_metadata.json for projecting range pixels later. '' to skip.")


    p.add_argument('--gt', nargs='*', default=[],
                   help='TUM trajectory file(s): one for all bags, or one per bag (same order as --bags).')
    p.add_argument('--max_gt_gap_ms', type=float, default=200.,
                   help='Leave the pose empty if no GT sample lies within this distance of the scan stamp.')

    p.add_argument('--ros2_distro', default='ROS2_HUMBLE',
                   help='Fallback message definitions for ROS2 bags without embedded types.')
    p.add_argument('--every', type=int, default=5, help='Keep every N-th scan (10 Hz scans are near-duplicates).')
    p.add_argument('--max_frames', type=int, default=0, help='Max frames per bag (0 = no limit).')

    # Normalization
    p.add_argument('--no_log', action='store_true', help='Skip the log1p compression.')
    p.add_argument('--pct', type=float, nargs=2, default=(1.0, 99.0), metavar=('LO', 'HI'),
                   help='Per-frame percentile clip over valid pixels.')
    p.add_argument('--clahe', action='store_true', help='Apply CLAHE after percentile normalization.')
    p.add_argument('--save_masks', action='store_true', help='Also save the valid-return mask.')
    return p.parse_args()


def resolve_bag_path(p):
    """ rosbags wants the ROS2 bag *directory*; accept a .db3/.mcap file inside it too. """
    p = Path(p)
    if p.suffix in ('.db3', '.mcap') and (p.parent / 'metadata.yaml').exists():
        return p.parent
    return p


def stamp_ns(msg, fallback):
    try:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
    except AttributeError:
        return fallback


################################ Decoding ################################

def image_to_numpy(msg):
    """ sensor_msgs/Image -> (H, W) numpy array in its native dtype. """
    enc = msg.encoding.lower()
    dtypes = {'mono8': np.uint8, '8uc1': np.uint8,
              'mono16': np.uint16, '16uc1': np.uint16,
              '32fc1': np.float32}
    if enc not in dtypes:
        raise ValueError(f'Unsupported image encoding: {msg.encoding}')
    dt = np.dtype(dtypes[enc]).newbyteorder('>' if msg.is_bigendian else '<')
    row = np.frombuffer(msg.data, dtype=dt).reshape(msg.height, msg.step // dt.itemsize)
    return row[:, :msg.width].astype(dt.newbyteorder('='))



################################ Normalization ################################

def normalize(signal, valid, args, clahe):
    """
        Map raw signal to uint8: 1 = no return, 2..255 = signal.
        0 is reserved: AugmentationPipe treats exact zeros as out-of-warp pixels.
    """
    x = signal.astype(np.float32)
    if not args.no_log:
        x = np.log1p(np.maximum(x, 0))

    out = np.ones(x.shape, np.uint8)
    if valid.sum() < 100:
        return out

    lo, hi = np.percentile(x[valid], args.pct)
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    x = (x * 253 + 2).astype(np.uint8)

    if clahe is not None:
        x = np.clip(clahe.apply(x), 2, 255).astype(np.uint8)

    out[valid] = x[valid]
    return out


################################ Ground truth ################################

class TumTrajectory:
    """ TUM file (t x y z qx qy qz qw) with linear position / slerp rotation interpolation. """

    def __init__(self, path):
        d = np.loadtxt(path)
        t = d[:, 0]
        _, first = np.unique(t, return_index=True)        # drop duplicate stamps
        d = d[np.sort(first)]
        d = d[np.argsort(d[:, 0])]
        self.t = d[:, 0]
        self.p = d[:, 1:4]
        self.slerp = Slerp(self.t, Rotation.from_quat(d[:, 4:8]))
        self.path = path

    def at(self, t, max_gap_s):
        """ Pose at time t (seconds) as [x y z qx qy qz qw], or None outside coverage. """
        if t < self.t[0] or t > self.t[-1]:
            return None, None
        i = np.searchsorted(self.t, t)
        gap = min(abs(t - self.t[max(i - 1, 0)]), abs(self.t[min(i, len(self.t) - 1)] - t))
        if gap > max_gap_s:
            return None, gap
        pos = np.array([np.interp(t, self.t, self.p[:, k]) for k in range(3)])
        return np.concatenate([pos, self.slerp([t]).as_quat()[0]]), gap


def gt_row(traj, s, args):
    if traj is None:
        return [''] * len(GT_COLS)
    pose, gap = traj.at(s * 1e-9, args.max_gt_gap_ms * 1e-3)
    if pose is None:
        return [''] * (len(GT_COLS) - 1) + ['' if gap is None else f'{gap * 1e3:.1f}']
    return [f'{v:.9f}' for v in pose] + [f'{gap * 1e3:.1f}']


################################ Extraction ################################

def list_topics(bags, typestore):
    for bag in bags:
        with AnyReader([resolve_bag_path(bag)], default_typestore=typestore) as reader:
            print(f'== {bag}')
            for c in sorted(reader.connections, key=lambda c: c.topic):
                print(f'  {c.topic:40s} {c.msgtype:40s} {c.msgcount}')


def save_metadata(reader, args, name):
    """ Dump the first Ouster metadata message (JSON string) next to the images. """
    conns = [c for c in reader.connections if c.topic == args.metadata_topic]
    if not conns:
        print(f'[warn] {name}: no {args.metadata_topic}; use the dataset calibration files '
              '(e.g. calibration_files/metadata_beta.json) for the beam model')
        return
    for conn, _, raw in reader.messages(connections=conns):
        with open(os.path.join(args.out, f'{name}_metadata.json'), 'w') as f:
            f.write(reader.deserialize(raw, conn.msgtype).data)
        return


class Saver:
    def __init__(self, args, writer, clahe, name, traj):
        self.args, self.writer, self.clahe, self.name, self.traj = args, writer, clahe, name, traj
        self.n_saved, self.n_posed = 0, 0

    def __call__(self, s, signal, valid):
        fname = f'{self.name}_{s}.png'
        cv2.imwrite(os.path.join(self.args.out, fname), normalize(signal, valid, self.args, self.clahe))
        if self.args.save_masks:
            cv2.imwrite(os.path.join(self.args.out, 'masks', fname), valid.astype(np.uint8) * 255)
        gt = gt_row(self.traj, s, self.args)
        self.n_posed += gt[0] != ''
        self.writer.writerow([fname, self.name, s] + gt)
        self.n_saved += 1



def extract_images(reader, args, save, name):
    use_range = bool(args.range_topic)
    topics = [args.signal_topic] + ([args.range_topic] if use_range else [])
    conns = [c for c in reader.connections if c.topic in topics]
    found = {c.topic for c in conns}
    if args.signal_topic not in found:
        print(f'[skip] {name}: no {args.signal_topic} (run with --list_topics)')
        return 0
    if use_range and args.range_topic not in found:
        print(f'[warn] {name}: no {args.range_topic}; using signal==0 as no-return')
        use_range = False
        conns = [c for c in conns if c.topic == args.signal_topic]

    n_signal = sum(c.msgcount for c in conns if c.topic == args.signal_topic)
    pending_sig, pending_rng = {}, {}   # stamp -> array, until both halves of a scan arrive
    n_seen = 0

    for conn, t, raw in tqdm.tqdm(reader.messages(connections=conns), desc=name,
                                  total=sum(c.msgcount for c in conns)):
        msg = reader.deserialize(raw, conn.msgtype)
        stamp = stamp_ns(msg, t)

        if conn.topic == args.signal_topic:
            n_seen += 1
            if (n_seen - 1) % args.every != 0:
                continue
            pending_sig[stamp] = image_to_numpy(msg)
        else:
            pending_rng[stamp] = image_to_numpy(msg)

        # Emit every scan that is complete
        for s in [s for s in pending_sig if (not use_range) or s in pending_rng]:
            sig = pending_sig.pop(s)
            if use_range:
                rng = pending_rng.pop(s)
                valid = (rng > 0) & (rng >= args.min_range)
            else:
                valid = sig > 0
            save(s, sig, valid)

        # Drop range images for scans we skipped, and anything stale
        if len(pending_rng) > 50:
            for s in sorted(pending_rng)[:-50]:
                pending_rng.pop(s)
        if len(pending_sig) > 50:
            print(f'[warn] {len(pending_sig)} signal frames without a matching range stamp; '
                  'check --range_topic or pass --range_topic ""')
            for s in sorted(pending_sig)[:-50]:
                pending_sig.pop(s)

        if args.max_frames and save.n_saved >= args.max_frames:
            break
    return n_signal


def extract_bag(bag, args, typestore, writer, clahe, traj):
    name = Path(bag).stem if Path(bag).is_file() else Path(bag).name
    save = Saver(args, writer, clahe, name, traj)

    with AnyReader([resolve_bag_path(bag)], default_typestore=typestore) as reader:
        if args.metadata_topic:
            save_metadata(reader, args, name)
        n_scans = extract_images(reader, args, save, name)

    msg = f'{name}: saved {save.n_saved} / {n_scans} scans'
    if traj is not None:
        msg += f', {save.n_posed} with GT pose from {os.path.basename(traj.path)}'
        if save.n_posed < save.n_saved:
            msg += f' ({save.n_saved - save.n_posed} outside GT coverage or gap > {args.max_gt_gap_ms:g} ms)'
    print(msg)
    return save.n_saved


def main():
    args = parse_args()
    typestore = get_typestore(getattr(Stores, args.ros2_distro))

    if args.list_topics:
        list_topics(args.bags, typestore)
        return
    if args.out is None:
        raise SystemExit('--out is required')
    if len(args.gt) not in (0, 1, len(args.bags)):
        raise SystemExit('--gt takes one file for all bags or one per bag')

    os.makedirs(args.out, exist_ok=True)
    if args.save_masks:
        os.makedirs(os.path.join(args.out, 'masks'), exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 16)) if args.clahe else None
    trajs = [TumTrajectory(g) for g in args.gt]

    index_path = os.path.join(args.out, 'index.csv')
    new_index = not os.path.exists(index_path)
    total = 0
    with open(index_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if new_index:
            writer.writerow(['file', 'bag', 'stamp_ns'] + GT_COLS)
        for i, bag in enumerate(args.bags):
            traj = trajs[0] if len(trajs) == 1 else (trajs[i] if trajs else None)
            total += extract_bag(bag, args, typestore, writer, clahe, traj)

    print(f'done: {total} images in {args.out}')
    if total < 3005:
        print(f'note: AugmentationPipe needs max_num_imgs + num_test_imgs images '
              f'(3000 + 5 by default); lower max_num_imgs for this dataset.')


if __name__ == '__main__':
    main()
