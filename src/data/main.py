import os
import boto3
from data import Data
from services.kafka import consumer, producer
from botocore.exceptions import ClientError
import json
import logging
from kafka import TopicPartition, OffsetAndMetadata
from kafka.errors import CommitFailedError
import concurrent.futures as cf
import threading
import shutil
import sys

BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVERS")
TOPIC_IN = os.getenv("TOPIC_IN")
TOPIC_OUT = os.getenv("TOPIC_OUT")
INPUT_DIR = os.getenv("S3_INPUT_DIR")
OUTPUT_DIR = os.getenv("S3_OUTPUT_DIR")
GROUP_ID = os.getenv("GROUP_ID")

print(f"TOPIC_IN:{TOPIC_IN}")
print(f"BOOTSTRAP_SERVER: {BOOTSTRAP_SERVERS}")
print(f"TOPIC_OUT:{TOPIC_OUT}")

kafka_consumer = consumer.Consumer(TOPIC_IN, BOOTSTRAP_SERVERS, GROUP_ID)
kafka_producer = producer.Producer(BOOTSTRAP_SERVERS, TOPIC_OUT)
print(f"Kafka Consumer initialized: {kafka_consumer}")
s3 = boto3.client('s3', region_name="ca-central-1")
print(f"S3: {s3}")

executor = cf.ThreadPoolExecutor(max_workers=4)
slots = threading.Semaphore(4)

in_flight = {}

def commit_exact(msg):
    tp = TopicPartition(msg.topic, msg.partition)
    off = OffsetAndMetadata(msg.offset + 1, None, leader_epoch=-1)
    try:
        kafka_consumer.consumer.commit({tp: off})   # commit *exactly* the msg you finished
    except CommitFailedError:
        # we were likely revoked; just continue—next poll() will rejoin
        pass

def process_one(msg):
    payload = msg.value
    key = json.loads(payload)["key"]
    local_tmp = f"/tmp/{key}"
    os.makedirs(os.path.dirname(local_tmp), exist_ok=True)
    try:
        s3.download_file(INPUT_DIR, key, local_tmp)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logging.warning("Missing S3 key %s; skipping and committing", key)
            commit_exact(msg)
        raise
    data = Data(local_tmp)
    data.process_video(f"/tmp", OUTPUT_DIR, key, s3)


def on_done(fut, tp, msg):
    try:
        fut.result()            
        commit_exact(msg)
        future = kafka_producer.producer.send(TOPIC_OUT, json.dumps({"video_name": key}).encode("utf-8"))
        try:
            md = future.get(timeout=25)
            print(f"[ack] {TOPIC_OUT}[{md.partition}]@{md.offset}"); sys.stdout.flush()
        except Exception as e:
            print(f"[error] send failed: {type(e).__name__}: {e}"); sys.stdout.flush()
            raise
        print(f"[ack] {TOPIC_OUT}[{md.partition}]@{md.offset}")
        print(f"Message sent to {TOPIC_OUT}")
        kafka_producer.producer.flush()

        try:
            shutil.rmtree(f"/tmp")
            print("Removed directory /tmp")
            shutil.rmtree(f"/app")
            print("Removed directory /app")
        except OSError:            
            pass

    except Exception as e:
        commit_exact(msg)
    
    finally:
        in_flight.pop(tp, None)
        kafka_consumer.consumer.resume(tp)
        slots.release()


print("Entering consume loop…")
while True:
    polled = kafka_consumer.consumer.poll(timeout_ms=200)
    if not polled:
        continue
    for partition, msgs in polled.items():
        for msg in msgs:
            # pause this partition so we don’t read ahead
            tp = TopicPartition(msg.topic, msg.partition)
            print("Partition: ", partition)
            if tp in in_flight:
                continue

            slots.acquire()

            payload = msg.value
            key = json.loads(payload)["key"]

            kafka_consumer.consumer.pause(tp)
            fut = executor.submit(process_one, msg)
            in_flight[tp] = fut

            fut.add_done_callback(lambda f, tp=tp, msg=msg: on_done(f, tp, msg))
