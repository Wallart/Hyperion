from datetime import datetime
from hyperion.utils.singleton import Singleton
from apscheduler.schedulers.background import BackgroundScheduler

import atexit


class TaskScheduler(metaclass=Singleton):
    def __init__(self):
        self._misfire_grace_time = 60 * 5  # 5 minutes late allowed
        self._scheduler = BackgroundScheduler()
        self._scheduler.start()
        # Shut down the scheduler when exiting the app
        atexit.register(lambda: self._scheduler.shutdown())

    def add_task(self, func, run_date: datetime):
        self._scheduler.add_job(func=func, trigger='date', run_date=run_date)

    def add_repeated_task(self, func, weeks=0, days=0, hours=0, minutes=0, seconds=0):
        kwargs = dict(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)
        self._scheduler.add_job(func=func, trigger='interval', **kwargs)
