from threading import Thread
from queue import Queue, PriorityQueue


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


class Sink:
    def __init__(self, queue):
        self._sink = queue
        self._timeout = 0.1

    def drain(self):
        job = self._sink.get(timeout=self._timeout)
        self._sink.task_done()
        return job


class Consumer(ThreadedTask):

    def __init__(self):
        super().__init__()
        self._in_queue = None
        self._timeout = 0.1

    def set_in_queue(self, queue):
        assert self._in_queue is None
        self._in_queue = queue

    def get_intake(self):
        return self._in_queue

    def create_intake(self, maxsize=0):
        queue = Queue(maxsize)
        self.set_in_queue(queue)
        return queue

    def _consume(self):
        job = self._in_queue.get(timeout=self._timeout)
        self._in_queue.task_done()
        return job


class Producer(ThreadedTask):

    def __init__(self):
        super().__init__()
        self._out_queues = []
        self._identified_out_queues = {}

    def pipe(self, consumer: Consumer):
        queue = Queue()
        self._out_queues.append(queue)
        consumer.set_in_queue(queue)
        return consumer

    def create_sink(self, maxsize=0):
        queue = Queue(maxsize)
        self._out_queues.append(queue)
        return Sink(queue)

    def create_identified_sink(self, identifier):
        # assert identifier not in self._identified_out_queues, 'Error identified queue already exists.'
        if identifier in self._identified_out_queues:
            return Sink(self._identified_out_queues[identifier])

        queue = PriorityQueue()
        self._identified_out_queues[identifier] = queue
        return Sink(queue)

    def set_identified_sink(self, identifier, sink):
        assert identifier not in self._identified_out_queues, 'Error identified queue already exists.'
        self._identified_out_queues[identifier] = sink._sink

    def delete_identified_sink(self, identifier):
        del self._identified_out_queues[identifier]

    # def get(self, identifier):
    #     assert identifier in self._identified_out_queues, 'Error identified queue not found.'
    #     return self._identified_out_queues[identifier]

    def _put(self, job, identifier):
        if identifier not in self._identified_out_queues:
            return False
        self._identified_out_queues[identifier].put(job)
        return True

    def _dispatch(self, job):
        _ = [q.put(job) for q in self._out_queues]
