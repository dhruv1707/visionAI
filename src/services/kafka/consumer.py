from kafka import KafkaConsumer
from kafka.sasl.oauth import AbstractTokenProvider
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
from kafka.admin import KafkaAdminClient, NewPartitions

class MSKTokenProvider(AbstractTokenProvider):
    def token(self):
        token, _ = MSKAuthTokenProvider.generate_auth_token('ca-central-1')
        return token

class Consumer():
    def __init__(self, topic, bootstrap_servers, group_id):
        token_provider = MSKTokenProvider()
        self.consumer = KafkaConsumer(topic, 
                                      bootstrap_servers=[bootstrap_servers],
                                      security_protocol='SASL_SSL',
                                      sasl_mechanism='OAUTHBEARER',
                                      sasl_oauth_token_provider=token_provider,
                                      max_poll_interval_ms=1000000,
                                      group_id=group_id,
                                      enable_auto_commit=False,
                                      auto_offset_reset='earliest',
                                      max_poll_records=1)
        
        admin_client = KafkaAdminClient(bootstrap_servers=[bootstrap_servers],
                                        security_protocol='SASL_SSL',
                                        sasl_mechanism='OAUTHBEARER',
                                        sasl_oauth_token_provider=token_provider)
        
        new_partition_count = 4
        new_partitions_request = {topic: NewPartitions(new_partition_count)}
        try:
            response = admin_client.create_partitions(new_partitions_request)
            print(f"Partition increase request sent: {response}")
        except Exception as e:
            print(f"Error increasing partitions: {e}")
        finally:
            admin_client.close()

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
