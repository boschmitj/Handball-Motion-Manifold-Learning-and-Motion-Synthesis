from dataclasses import dataclass

@dataclass
class Capture:
    def __init__(
        self,
        capture_id: int,
        capture_name: str,
        capture_path: str,
        json_path: str
    ):
        self.capture_id = capture_id
        self.capture_name = capture_name
        self.capture_path = capture_path
        self.json_path = json_path

    def serialize(self):
        return {
            "captureId": self.capture_id,
            "captureName": self.capture_name,
            "capturePath": self.capture_path,
            "jsonPath": self.json_path,
        }
