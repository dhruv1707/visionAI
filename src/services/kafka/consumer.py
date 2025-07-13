from kafka import KafkaConsumer

class Consumer():
    def __init__(self, topic, bootstrap_servers):
        self.consumer = KafkaConsumer(topic, 
                                      bootstrap_servers=bootstrap_servers,
                                      auto_offset_reset='earliest')

    def consume(self):
        print("Starting to consume...")
        print(self.consumer)
        try:
            for msg in self.consumer:
                print(f"Received: {msg.value!r} "
            f"(partition {msg.partition}, offset {msg.offset})")
        finally:
            self.consumer.close()
            print("Consumer closed.")
