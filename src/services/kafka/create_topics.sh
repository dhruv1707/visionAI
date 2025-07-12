#!/usr/bin/env bash
set -e

KAFKA_BIN=/opt/kafka/bin

$KAFKA_BIN/kafka-topics.sh --bootstrap-server 192.168.2.96:9092 --create --topic video_uploads --if-not-exists

echo "▶ Created topic: video_uploads"

$KAFKA_BIN/kafka-topics.sh --bootstrap-server 192.168.2.96:9092 --create --topic chunks_ready --if-not-exists

echo "▶ Created topic: chunks_ready"

echo "✅ All topics created (or already existed)"