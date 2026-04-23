import time

class PerformanceTracker:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0

    def update(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        return fps
        
    def reset(self):
        self.start_time = time.time()
        self.frame_count = 0