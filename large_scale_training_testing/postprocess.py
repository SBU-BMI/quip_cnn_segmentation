from multiprocessing import Process, Queue
from watershed.seg_with_watershed import apply_segmentation


def worker(id, task_params):
    apply_segmentation(**task_params)


def queue_waiter(id, queue):
    while True:
        task_params = queue.get()
        if task_params is None:
            break
        worker(id, task_params)
    queue.put(None)


class MultiProcWatershed:
    def __init__(self, n_proc):
        self.queue = Queue()
        self.n_proc = n_proc

    def start(self):
        self.workers = [Process(target=queue_waiter, args=(i, self.queue,))
                        for i in xrange(self.n_proc)]
        for w in self.workers:
            w.start()

    def add_job(self, task_params):
        self.queue.put(task_params)

    def wait_til_stop(self):
        self.queue.put(None)
        for i in range(self.n_proc):
            self.workers[i].join()
        self.queue.close()

