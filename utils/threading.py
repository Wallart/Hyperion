from queue import Queue
from threading import Thread


class ThreadedTask(Thread):
    def __init__(self):
        super().__init__()
        # Daemon thread are killed abruptly when main thread is dead
        self.daemon = False
        self.running = False

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False


class Consumer(ThreadedTask):

    def __init__(self):
        super().__init__()
        self._in_queue = None

    def set_in_queue(self, queue):
        self._in_queue = queue

    def create_intake(self):
        queue = Queue()
        self.set_in_queue(queue)
        return queue


class Producer(ThreadedTask):

    def __init__(self):
        super().__init__()
        self._out_queues = []

    def pipe(self, consumer: Consumer):
        queue = Queue()
        self._out_queues.append(queue)
        consumer.set_in_queue(queue)
        return consumer

    def create_sink(self):
        queue = Queue()
        self._out_queues.append(queue)
        return queue

    def _dispatch(self, job):
        _ = [q.put(job) for q in self._out_queues]
