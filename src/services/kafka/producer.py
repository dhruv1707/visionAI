from kafka import KafkaProducer
from kafka.sasl.oauth import AbstractTokenProvider
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
from kafka.admin import KafkaAdminClient, NewTopic

class MSKTokenProvider(AbstractTokenProvider):
    def token(self):
        token, _ = MSKAuthTokenProvider.generate_auth_token('ca-central-1')
        return token

def create_topic(bootstrap_servers, topic_name, token_provider, num_partitions=1, replication_factor=1):
    
    print(f"Creating topic: {topic_name}")
    admin_client = None
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=[bootstrap_servers],
            security_protocol='SASL_SSL',
            sasl_mechanism='OAUTHBEARER',
            sasl_oauth_token_provider=token_provider
        )

        existing_topics = admin_client.list_topics()
        if topic_name in existing_topics:
            print(f"Topic: {topic_name} already exists")
            return

        new_topic = NewTopic(name=topic_name, num_partitions=num_partitions, replication_factor=replication_factor)
        admin_client.create_topics(new_topics=[new_topic], validate_only=False)
        print(f"Topic '{topic_name}' created successfully.")
        
    except Exception as e:
        print(f"Error creating topic '{topic_name}': {e}")
    finally:
        if admin_client:
            admin_client.close()


class Producer():
    def __init__(self, bootstrap_servers, topic):
        token_provider = MSKTokenProvider()

        self.producer = KafkaProducer(bootstrap_servers=[bootstrap_servers],
                                      security_protocol='SASL_SSL',
                                      sasl_mechanism='OAUTHBEARER',
                                      sasl_oauth_token_provider=token_provider,
                                      retries=5)
        
        create_topic(bootstrap_servers, topic, token_provider) 

        print(f"Kafka producer connected: {self.producer}")  

    def produce(self, topic, msg):
        self.producer.send(topic, msg)
        print(f"Message sent: {msg}")
        self.producer.flush()
        self.producer.close()

