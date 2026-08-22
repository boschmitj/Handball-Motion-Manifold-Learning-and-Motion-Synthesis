from dataclasses import dataclass

@dataclass
class Task:
    def __init__(
        self,
        task_id: int,
        capture_id: int,
        username: str,
        created_at: str
    ):
        self.task_id = task_id
        self.capture_id = capture_id
        self.username = username
        self.created_at = created_at

    def serialize(self):
        return {
            "taskId": self.task_id,
            "captureId": self.capture_id,
            "username": self.username,
            "createdAt": self.created_at,
        }
