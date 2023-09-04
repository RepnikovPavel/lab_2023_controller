import time


class Timer:
    execution_time = 0.0

    start_of_period = 0.0

    def start(self):
        self.execution_time = 0.0
        self.execution_time = time.time()

    def start_time_period(self):
        self.start_of_period = time.time()

    def stop_time_period(self):
        return time.time() - self.start_of_period

    def stop(self):
        self.execution_time = time.time() - self.execution_time

    def get_execution_time(self):
        return self.execution_time

    def print_execution_time(self):
        print('elapsed time {} sek'.format(self.get_execution_time()))
