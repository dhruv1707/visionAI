import os
import boto3
from services.kafka import consumer, producer
import json
import datetime
from kafka import TopicPartition, OffsetAndMetadata
from kafka.errors import CommitFailedError
import time


BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVERS")
TOPIC_IN = os.getenv("TOPIC_IN")
INPUT_DIR = os.getenv("S3_INPUT_DIR")
OUTPUT_DIR = os.getenv("S3_OUTPUT_DIR")
GROUP_ID = os.getenv("GROUP_ID")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
TOPIC_OUT = os.getenv("TOPIC_OUT")
BUCKET = os.getenv("BUCKET")


print(f"TOPIC_IN:{TOPIC_IN}")
print(f"BOOTSTRAP_SERVER: {BOOTSTRAP_SERVERS}")

kafka_consumer = consumer.Consumer(TOPIC_IN, BOOTSTRAP_SERVERS, GROUP_ID)
kafka_producer = producer.Producer(BOOTSTRAP_SERVERS, TOPIC_OUT)
print(f"Kafka Consumer initialized: {kafka_consumer}")
print(f"Kafka Producer initialized: {kafka_producer}")
s3 = boto3.client('s3', region_name="ca-central-1")
print(f"S3: {s3}")
runtime = boto3.client('sagemaker-runtime')

kafka_consumer.consumer.poll(timeout_ms=0)
ass = list(kafka_consumer.consumer.assignment())
print("ASSIGNMENT:", ass)

if not ass:
    print("TOPICS ON CLUSTER:", kafka_consumer.consumer.topics())  # must contain TOPIC_IN
    print("PARTITIONS FOR TOPIC:", kafka_consumer.consumer.partitions_for_topic(TOPIC_IN))
    kafka_consumer.consumer.unsubscribe()
    tps = [TopicPartition(TOPIC_IN, p) for p in kafka_consumer.consumer.partitions_for_topic(TOPIC_IN)]
    kafka_consumer.consumer.assign(tps)
    print("ASSIGNMENT after manual assign:", list(kafka_consumer.consumer.assignment()))

def call_sagemaker(chunk_folder):
    payload = {"path_to_data": chunk_folder}
    response = runtime.invoke_endpoint(
        EndpointName = ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    return json.loads(response["Body"].read())["summaries"]

def list_chunks(user_id, video_name):
    prefix = f"processed-videos-visionai/{user_id}/{video_name}/"
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return [p["Prefix"] for p in response.get("CommonPrefixes", [])]


def list_content_in_prefix(prefix):
    response = s3.list_objects_v2(Bucket=BUCKET,Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]

def upload_captions_to_s3(chunk_name, captions):
    key = f"summaries/{chunk_name}.json"
    body = json.dumps({
        "chunk_name": chunk_name,
        "captions": captions,
        "processed_at": datetime.utcnow().isoformat() + "Z"
    })
    s3.put_object(
        Bucket=OUTPUT_DIR, Key=key, Body=body,
        ContentType="application/json"
    )
    return {"chunk_name": chunk_name, "summaries_key": key, "processed_at": datetime.utcnow().isoformat() + "Z"}

def commit_exact(msg):
    tp = TopicPartition(msg.topic, msg.partition)
    off = OffsetAndMetadata(msg.offset + 1, None, leader_epoch=-1)
    try:
        kafka_consumer.consumer.commit({tp: off})   # commit *exactly* the msg you finished
    except CommitFailedError:
        # we were likely revoked; just continue—next poll() will rejoin
        pass


# while not kafka_consumer.consumer.assignment():
#     kafka_consumer.consumer.poll(timeout_ms=200)

# kafka_consumer.consumer.poll(timeout_ms=0)
# tps = list(kafka_consumer.consumer.assignment())
# ends = kafka_consumer.consumer.end_offsets(tps)
# print("Tps: ", tps, "ends: ", ends)

# for tp in tps:
#     pos = kafka_consumer.consumer.position(tp)   # None until we actually fetch
#     print(f"{tp}: pos={pos} end={ends[tp]}")

print("Entering consume loop…")
while True:
    polled = kafka_consumer.consumer.poll(timeout_ms=1500)
    if not polled:
        print("no messages yet…"); time.sleep(1); continue
    print("Frame_summary service polled for messages: ", polled)
    for partition, msgs in polled.items():
        print("Partition: ", partition, "Msgs: ", msgs)
        for msg in msgs:
            tp = TopicPartition(msg.topic, msg.partition)
            print("Partition: ", partition)
            kafka_consumer.consumer.pause(tp)
            try:

                payload = msg.value
                print(f"Payload: {payload}")
                user_id = json.loads(payload)["user_id"]
                video_name = json.loads(payload)["video_name"]
                local_tmp = f"/tmp/"
                os.makedirs(os.path.dirname(local_tmp), exist_ok=True)
                print(f"msg inside for loop: {msg}")

                for chunk_prefix in list_chunks(user_id, video_name):

                    keys = list_content_in_prefix(chunk_prefix)
                    frame_keys = [k for k in keys if k.endswith(".png")]
                    transcript_keys = [k for k in keys if k.endswith(".txt")]

                    local_files = []
                    for key in frame_keys + [transcript_keys]:
                        local_path = os.path.join("/tmp", os.path.basename(key))
                        s3.download_file(INPUT_DIR, key, local_path)
                        local_files.append(local_path)

                captions = call_sagemaker("/tmp")
                chunk_name = os.path.basename(chunk_prefix.strip("/"))
                payload = upload_captions_to_s3(chunk_name, captions)

                commit_exact(msg)
                kafka_producer.producer.send(TOPIC_OUT, json.dumps({payload}).encode('utf-8'))
                print(f"Message sent to {TOPIC_OUT}")
                kafka_producer.producer.flush()
            finally:
                kafka_consumer.consumer.resume(tp)


