from dataclasses import dataclass

@dataclass
class Sequence:
    def __init__(
        self,
        sequence_id: int,
        capture_id: int,
        sequence_name: str,
        sequence_path: str,
    ):
        self.sequence_id = sequence_id
        self.capture_id = capture_id
        self.sequence_name = sequence_name
        self.sequence_path = sequence_path

    def serialize(self):
        return {
            "sequenceId": self.sequence_id,
            "captureId": self.capture_id,
            "sequenceName": self.sequence_name,
            "sequencePath": self.sequence_path,
        }

def get_sequences_from_data(data):
    # Accept both shapes used in the wild: the legacy `ioi` key
    # (user-imported captures) and the `name` key produced by
    # the released-dataset capture generator for released datasets.
    sequences_data = data.get('sequences', {}) or {}
    sequences_list = []
    for _, val in sequences_data.items():
        if not isinstance(val, dict):
            continue
        name = val.get('name') or val.get('ioi')
        if name:
            sequences_list.append(name)
    sequences_list.sort()
    return sequences_list
