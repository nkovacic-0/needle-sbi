"""
Multiprocessing-safe data reader
"""

import logging
import multiprocessing as mp

import dask
import dask_awkward as dak

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

def load_partition(
    array: dak.Array,  # type: ignore
    partition_id: int,
    event_index: int = None,
) -> dak.Array:  # type: ignore
    """Delayed read of a dask partition

    Args:
        array (dask_awkward.Array): Delayed Array
        partition_id (int): The index of the desired partition
        event_index (int, Optional): The *local* index to slice the partition with.
            Cases:
            - None: Returns the whole partition
            - int > 0: Returns all events up to that index
            - int < 0: Returns all events above this index

    Returns:
        dask_awkward.Array:
            Delayed partition as a dak.Array.
    """
    logger.debug(f"Loading partition {partition_id}.")
    if not event_index:
        return array.partitions[partition_id]
    if event_index > 0:
        return array.partitions[partition_id][0:event_index]
    if event_index < 0:
        return array.partitions[partition_id][abs(event_index) : ]


class PartitionQueue:
    """
    Class that manages the reading of individual dask partitions in a thread-safe manner.
    This is specially useful for using multi-processing from libraries other than dask,
    such as pytorch's DataLoader. The read Lock ensures that only one thread can read the
    file at a time.
    """

    array: dak.Array  # type: ignore
    total_num_partitions: int

    def __init__(
        self,
        array: dak.Array,  # type: ignore
    ) -> None:
        """Initialize a PartitionQueue instance for the given dask awkward array.

        Args:
            array (dak.Array): Dask Awkward Array to manage partitions.
        """
        # NOTE: REQUIRED: dask.config.set here forces dask's default (threaded) 
        # scheduler off globally. Without this, any parent-process dask.compute() 
        # call (e.g. a Scaler fitting in DataModule.setup(), which runs before 
        # DataLoader forks workers) can lazily create dask's global ThreadPoolExecutor. 
        # Forking a process with live threads is unsafe — a lock held by a non-forking 
        # thread at fork time is copied into the child in a permanently-locked state, 
        # since only the forking thread survives fork(). This causes a silent, total 
        # deadlock the moment a forked DataLoader worker calls .compute() itself. 
        # Confirmed via direct testing:
        # removing this line reproduces a persistent, zero-progress hang under
        # num_workers>0, while restoring it resolves it.
        # Also tested: this line used to be in class declaration, but moving it to
        # init didn't cause any issues in execution of the torch multithreading pipeline
        # In case such issues creep up again, try moving dask.config.set up to class dec 
        dask.config.set( scheduler="single-threaded", num_workers=1, )

        self.array = array
        self.total_num_partitions = array.npartitions

        if self.total_num_partitions > 1:
            if not any(self.array.divisions):
                self.array.eager_compute_divisions()
            self.read_lock = mp.Lock()
            # NOTE: removed this line to investigate a hang on multithreading
            # self.manager = mp.Manager()

    def load_partition_thread_safe(
        self,
        partition_id: int,
        event_index: int | None,
    ) -> dak.Array:
        """Load a partition from the Dask Awkward Array.

        Args:
            partition_id (int): Index of the partition to load.
            event_index (int, Optional): The *local* index to slice the Array with. See documentation
                of :func:`load_partition` for more information.

        Returns:
            ak.Array: The loaded partition as an Awkward Array. Will execute the dask graph
                using the '.compute()' method. Will always return a single partition.
        """
        if self.total_num_partitions == 1:
            return self.array

        with self.read_lock:
            return load_partition(self.array, partition_id, event_index)
