import os
import time
import fnmatch
from multiprocessing import Process, Queue
from watershed.seg_with_watershed import apply_segmentation
from ctypes import c_char_p


def worker(id, task_params):
    apply_segmentation(**task_params)


def queue_waiter(id, queue):
    while True:
        task_params = queue.get()
        if task_params is None:
            break
        worker(id, task_params)
    queue.put(None)


def finish_checker(finishing_queue):
    def not_finished(folder):
        n_seg = len(fnmatch.filter(os.listdir(folder), '*_SEG.png'))
        n_json = len(fnmatch.filter(os.listdir(folder), '*-algmeta.json'))
        if n_seg == n_json:
            return False
        return True

    def touch(fname, times=None):
        with open(fname, 'a'):
            os.utime(fname, times)

    while True:
        folder = finishing_queue.get()
        if folder is None:
            return
        elif not_finished(folder):
            finishing_queue.put(folder)
            time.sleep(1)
        else:
            touch(os.path.join(folder, 'finish'))
            print 'Segmentation finished under {}'.format(folder)


class MultiProcWatershed:
    def __init__(self, n_proc):
        self.queue = Queue()
        self.finishing_queue = Queue()
        self.n_proc = n_proc

    def start(self):
        self.workers = [Process(target=queue_waiter, args=(i, self.queue,))
                        for i in xrange(self.n_proc)]
        for w in self.workers:
            w.start()

        self.wsi_checker = Process(target=finish_checker, args=(self.finishing_queue,))
        self.wsi_checker.start()

    def add_job(self, task_params):
        self.queue.put(task_params)

    def add_finisher(self, folder):
        self.finishing_queue.put(folder)

    def wait_til_stop(self):
        self.queue.put(None)
        for i in range(self.n_proc):
            self.workers[i].join()

        # Wait a bit to let all finish files generated
        time.sleep(20)
        self.finishing_queue.put(None)
        self.wsi_checker.join()

        self.queue.close()
        self.finishing_queue.close()

