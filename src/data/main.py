import os
import boto3
from data import Data
from services import kafka

BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVER")
TOPIC_IN = os.getenv("TOPIC_IN")
INPUT_DIR = os.getenv("S3_INPUT_BUCKET")
OUTPUT_DIR = os.getenv("S3_OUTPUT_DIR")
GROUP_ID = os.getenv("GROUP_ID")

consumer = kafka.consumer(TOPIC_IN, BOOTSTRAP_SERVERS)
s3 = boto3.client('s3')

for msg in consumer:
    payload = msg.value
    key = payload["key"]
    local_tmp = f"/tmp"
    s3.download_file(INPUT_DIR, key, local_tmp)
    data = Data(local_tmp)
    data.process_video(local_tmp, OUTPUT_DIR)
