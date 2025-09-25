from kafka import KafkaConsumer
from kafka.sasl.oauth import AbstractTokenProvider
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider

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
                                      auto_offset_reset='latest',
                                      max_poll_records=1)
        
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
