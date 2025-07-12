from producer import Producer
from consumer import Consumer
import time

kafka_producer = Producer()
kafka_producer.produce("chunks_ready", b"test_msg")

time.sleep(2)

kafka_consumer = Consumer("chunks_ready")
kafka_consumer.consume()