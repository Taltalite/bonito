import unittest
from types import SimpleNamespace
from unittest import mock

import bonito.io
import bonito.pod5


class FakeRunInfoTable:
    def read_pandas(self):
        row = SimpleNamespace(
            tracking_id={"run_id": "run1", "exp_start_time": "2024-01-01T00:00:00Z"},
            flow_cell_id="fc1",
            system_name="sys1",
            sample_id="sample1",
        )
        return SimpleNamespace(itertuples=lambda: iter([row]))


class FakePod5Handle:
    def __init__(self, reads):
        self._reads = list(reads)
        self.run_info_table = FakeRunInfoTable()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def reads(self, preload=None):
        return iter(self._reads)

    def read_batches(self):
        return [SimpleNamespace(num_reads=len(self._reads))]


class FakeRead:
    def __init__(self, read_id):
        self.read_id = read_id


class TestPod5ReadFiltering(unittest.TestCase):
    def setUp(self):
        self.handles = {
            "a.pod5": FakePod5Handle([FakeRead("r1"), FakeRead("r2")]),
            "b.pod5": FakePod5Handle([FakeRead("r3")]),
        }

    def _reader_factory(self, pod5_file):
        return self.handles[str(pod5_file)]

    def test_pod5_reads_filters_requested_ids_in_python(self):
        with mock.patch("bonito.pod5.Reader", side_effect=self._reader_factory):
            reads = list(bonito.pod5.pod5_reads("a.pod5", {"r2"}))
        self.assertEqual([str(read.read_id) for read in reads], ["r2"])

    def test_get_read_groups_counts_only_filtered_reads(self):
        with mock.patch("bonito.pod5.Reader", side_effect=self._reader_factory), \
             mock.patch("bonito.pod5.glob", return_value=["a.pod5", "b.pod5"]):
            groups, num_reads = bonito.pod5.get_read_groups(
                directory="unused",
                model="model",
                read_ids={"r2"},
                recursive=False,
            )
        self.assertEqual(num_reads, 1)
        self.assertEqual(len(groups), 1)


class DummyAlignmentFile:
    def __init__(self, *args, **kwargs):
        self.header = kwargs.get("header")

    def write(self, *args, **kwargs):
        return None


class DummyAlignmentHeader:
    @staticmethod
    def from_references(*args, **kwargs):
        return object()


class DummyCSVLogger:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def append(self, row):
        return None


class FakeWriterRead:
    def __init__(self):
        self.filename = "fake.pod5"
        self.signal = [0.0, 1.0]
        self.read_id = "read-1"
        self.num_samples = 2
        self.trimmed_samples = 0
        self.run_id = "run1"
        self.channel = 1
        self.mux = 1
        self.start = 0.0
        self.duration = 1.0
        self.template_start = 0.0
        self.template_duration = 1.0

    def tagdata(self):
        return []


class DummyAlignedSegment:
    @staticmethod
    def fromstring(*args, **kwargs):
        return object()


class TestWriterShortSequenceWarning(unittest.TestCase):
    def test_writer_warns_on_length_one_sequence(self):
        read = FakeWriterRead()
        result = {
            "sequence": "A",
            "qstring": "!",
            "moves": None,
            "mapping": False,
        }
        with mock.patch("bonito.io.AlignmentFile", DummyAlignmentFile), \
             mock.patch("bonito.io.AlignmentHeader", DummyAlignmentHeader), \
             mock.patch("bonito.io.CSVLogger", DummyCSVLogger), \
             mock.patch("bonito.io.sam_record", return_value="record"), \
             mock.patch("bonito.io.AlignedSegment", DummyAlignedSegment), \
             mock.patch.object(bonito.io.logger, "warning") as warning:
            writer = bonito.io.Writer(
                "wb",
                iter([(read, result)]),
                aligner=None,
                group_key="model",
                groups=[],
                min_qscore=0,
            )
            writer.run()

        warning.assert_called_once()
        self.assertIn("suspiciously short sequence length 1", warning.call_args[0][0])
