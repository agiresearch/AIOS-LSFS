from .base import BaseDb
import redis

class Redis(BaseDb):
    def __init__(self) -> None:
        super().__init__()
        self.client = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
        
    def add_file(self):
        pass
    
    def update_file(self):
        pass
    
    def read_file(self):
        pass
    
    def delete_file(self):
        pass