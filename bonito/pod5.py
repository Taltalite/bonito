"""
Bonito POD5 Utils
"""

from glob import glob
from pathlib import Path
from collections import OrderedDict
from datetime import timedelta, timezone

import numpy as np
import bonito.reader
from pod5 import Reader


class Read(bonito.reader.Read):

    def __init__(self, read, filename, meta=False, do_trim=True, scaling_strategy=None, norm_params=None):

        self.meta = meta

        self.read_id = read.read_id
        self.run_info = read.run_info
        self.filename = filename.name

        self.sample_id = self.run_info.sample_id
        self.run_id = self.run_info.acquisition_id
        self.acquisition_start_time = self.run_info.acquisition_start_time
        self.exp_start_time = self.acquisition_start_time.isoformat().replace('Z', '')

        self.flow_cell_id = self.run_info.flow_cell_id
        self.device_id = self.run_info.sequencer_position

        if self.meta:
            return

        self.pore = read.pore
        self.mux = self.pore.well
        self.channel = self.pore.channel
        self.read_number = read.read_number
        self.num_samples = read.sample_count

        self.context_tags = dict(self.run_info.context_tags)
        self.sample_rate = int(self.context_tags['sample_frequency'])

        self.start = read.start_sample / self.sample_rate
        self.duration = self.num_samples / self.sample_rate

        start_time = self.acquisition_start_time + timedelta(seconds=self.start)
        self.start_time = start_time.astimezone(timezone.utc).isoformat(timespec="milliseconds")

        self.raw = read.signal

        self.calibration = read.calibration
        self.scaling = self.calibration.scale
        self.offset = self.calibration.offset
        self.scaled = self.scaling * (self.raw.astype(np.float32) + self.offset)
        self.scaling_strategy = ("quantile" if scaling_strategy is None else
                                 scaling_strategy.get("strategy","quantile")) 
        
        self.shift, self.scale = bonito.reader.normalisation(self.scaled, scaling_strategy, norm_params)
        self.trimmed_samples = bonito.reader.trim(self.scaled, threshold=self.scale * 2.4 + self.shift) if do_trim else 0

        self.template_start = self.start + (self.trimmed_samples / self.sample_rate)
        self.template_duration = self.duration - (self.trimmed_samples / self.sample_rate)

        self.signal = (self.scaled[self.trimmed_samples:] - self.shift) / self.scale


def _normalize_read_ids(read_ids):
    if read_ids is None:
        return None
    return {str(read_id) for read_id in read_ids}


def pod5_reads(pod5_file, read_ids, skip=False, preload=None):
    """
    Get all the reads from the `pod5_file`.
    """
    wanted = _normalize_read_ids(read_ids)
    preload = ["samples"] if preload is None else preload

    if wanted is None and not skip:
        yield from Reader(pod5_file).reads(preload=preload)
        return

    for read in Reader(pod5_file).reads(preload=preload):
        read_id = str(read.read_id)
        if wanted is None:
            keep = True
        else:
            keep = read_id in wanted
            if skip:
                keep = not keep
        if keep:
            yield read


def get_read_groups(directory, model, read_ids=None, skip=False, n_proc=1, recursive=False, cancel=None):
    """
    Get all the read meta data for a given `directory` using the pod5 run info table
    """
    groups = set()
    num_reads = 0
    pattern = "**/*.pod5" if recursive else "*.pod5"
    pod5_files = (Path(x) for x in glob(directory + "/" + pattern, recursive=True))

    for pod5_file in pod5_files:
        with Reader(pod5_file) as pod5_fh:
            if read_ids is None and not skip:
                matched_reads = sum(batch.num_reads for batch in pod5_fh.read_batches())
            else:
                matched_reads = sum(1 for _ in pod5_reads(pod5_file, read_ids, skip, preload=[]))

            if matched_reads == 0:
                continue

            num_reads += matched_reads
            for run_info_row in pod5_fh.run_info_table.read_pandas().itertuples():
                tracking = dict(run_info_row.tracking_id)
                groupdict = OrderedDict([
                    ('ID', f"{tracking['run_id']}_{model}"),
                    ('PL', "ONT"),
                    ('DT', f"{tracking['exp_start_time']}"),
                    ('PU', f"{run_info_row.flow_cell_id}"),
                    ('PM', f"{run_info_row.system_name}"),
                    ('LB', f"{run_info_row.sample_id}"),
                    ('SM', f"{run_info_row.sample_id}"),
                    ('DS', f"run_id={tracking['run_id']} "
                     f"basecall_model={model}")
                ])
                groups.add('\t'.join(["@RG", *[f"{k}:{v}" for k, v in groupdict.items()]]))
    return groups, num_reads


def get_reads(directory, read_ids=None, skip=False, n_proc=1, recursive=False, do_trim=True, scaling_strategy=None, norm_params=None, cancel=None):
    """
    Get all reads in a given `directory`.
    """
    pattern = "**/*.pod5" if recursive else "*.pod5"
    pod5_files = (Path(x) for x in glob(directory + "/" + pattern, recursive=True))

    for pod5_file in pod5_files:
        for read in pod5_reads(pod5_file, read_ids, skip):
            yield Read(read, pod5_file, do_trim=do_trim, scaling_strategy=scaling_strategy, norm_params=norm_params)
            if cancel is not None and cancel.is_set():
                return
