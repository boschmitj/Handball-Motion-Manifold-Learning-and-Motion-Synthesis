from pathlib import Path
import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('HOST')
USER = os.getenv('USER')
PASSWORD = os.getenv('PASSWORD')
DOMAIN = os.getenv('DOMAIN')


REMOTE_DIR = (
    f"/home/{DOMAIN}/{USER}/Handballdaten_BA/"
    "tb15qabi-kinexon-xg-data-2023_24/"
    "tb15qabi-kinexon-xg-data-2023_24/"
    "tb15qabi-kinexon-xg-data-2023_24_package01/"
    "HBL_2023_2024/"
)
LOCAL_DIR = Path("./games_position_files")
FILE_SUFFIX = "_2_phases_positions.csv"

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(
    HOST,
    username=USER,
    password=PASSWORD,
)

sftp = ssh.open_sftp()

try:
    remote_files = sorted(sftp.listdir(REMOTE_DIR))
    matching_files = [name for name in remote_files if name.endswith(FILE_SUFFIX)]

    # Filter out files that already exist locally
    existing_files = set(f.name for f in LOCAL_DIR.glob(f"*{FILE_SUFFIX}"))
    files_to_download = [name for name in matching_files if name not in existing_files]
    already_present = len(matching_files) - len(files_to_download)

    print(f"Found {len(matching_files)} position files (remote)")
    print(f"  {already_present} already present locally")
    print(f"  {len(files_to_download)} to download")

    for index, filename in enumerate(files_to_download, start=1):
        remote_path = f"{REMOTE_DIR}{filename}"
        local_path = LOCAL_DIR / filename
        print(f"[{index}/{len(files_to_download)}] Downloading {filename}")
        sftp.get(remote_path, str(local_path))

    print(
        f"Download complete: {len(files_to_download)} new files saved to {LOCAL_DIR}, "
        f"total now: {already_present + len(files_to_download)}"
    )
finally:
    sftp.close()
    ssh.close()