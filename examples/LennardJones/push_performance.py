#!/usr/bin/env python3

from datetime import datetime
import json
from pathlib import Path
import subprocess

files = list(Path('.').glob('*.json'))
assert (len(files) == 1)

with open(files[0], 'r') as fin:
    data = json.load(fin)
data['datetime'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data['git'] = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import os

# You can generate a Token from the "Tokens Tab" in the UI
token = os.environ["INFLUXDB_TOKEN"]
org = os.environ["INFLUXDB_ORG"]
bucket = os.environ["INFLUXDB_BUCKET"]

client = InfluxDBClient(url=os.environ["INFLUXDB_URL"], token=token)

point = Point("performance") \
    .tag("git", data['git']) \
    .field("total-app-time", data['kokkos-kernel-data']['total-app-time']) \
    .field("total-kernel-times", data['kokkos-kernel-data']['total-kernel-times']) \
    .field("total-non-kernel-times", data['kokkos-kernel-data']['total-non-kernel-times']) \
    .field("percent-in-kernels", data['kokkos-kernel-data']['percent-in-kernels']) \
    .field("unique-kernel-calls", data['kokkos-kernel-data']['unique-kernel-calls']) \
    .time(datetime.utcnow(), WritePrecision.NS)

write_api = client.write_api(write_options=SYNCHRONOUS)
write_api.write(bucket, org, point)
