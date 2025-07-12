from kafka import KafkaProducer

class Producer():
    def __init__(self):
        self.producer = KafkaProducer(bootstrap_servers=['192.168.2.96:9092'],
                                      retries=5) 

        print("Producer connected")  

    def produce(self, topic, msg):
        self.producer.send(topic, msg)
        print(f"Message sent: {msg}")
        self.producer.flush()
        self.producer.close()

