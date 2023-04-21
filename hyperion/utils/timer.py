from time import time
from hyperion.utils import Singleton
from hyperion.utils.logger import ProjectLogger

import ntplib


class Timer(metaclass=Singleton):

    def __init__(self, server='pool.ntp.org'):
        self._server_url = server
        self._client = ntplib.NTPClient()
        self._time_difference = 0

        self.update_reference_time()

    def update_reference_time(self, retries=10):
        try:
            t0 = time()
            response = self._client.request(self._server_url)
            response_time = time() - t0
            if response_time > .25 and retries > 0:
                ProjectLogger().warning('NTP request was too slow. Retrying...')
                self.update_reference_time(retries - 1)
                return

            ntp_time = response.tx_time
            self._time_difference = ntp_time - t0
        except Exception as e:
            retry = retries - 1
            message = 'You might expect desync behaviors.' if retries == 0 else f'Retry {10 - retry}/10'
            ProjectLogger().warning(f'Cannot communicate with NTP server. {message}')
            if retries > 0:
                self.update_reference_time(retry)

    def now(self):
        return time() + self._time_difference

    @staticmethod
    def gt(timestamp, ref_time):
        if (timestamp - ref_time) <= -.25:
            return False
        return True


if __name__ == '__main__':
    timer = Timer()
    print(timer.now())
