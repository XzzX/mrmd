#!/usr/bin/env python3
# Copyright 2024 Sebastian Eibl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(
    description="Parse Kokkos performance results and send them to InfluxDB"
)
parser.add_argument("type", type=str, help="simulation type")
args = parser.parse_args()

files = list(Path(".").glob("*0.json"))
assert len(files) == 1
file = files[0]

from kokkos_tools import print_kernel_runtimes

print_kernel_runtimes(str(file), 10)

with open(str(file), "r") as fin:
    data = json.load(fin)
data["datetime"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data["git"] = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
data["type"] = args.type
file.rename(f"{args.type}.json")

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import os

# You can generate a Token from the "Tokens Tab" in the UI
token = os.environ["INFLUXDB_TOKEN"]
org = os.environ["INFLUXDB_ORG"]
bucket = os.environ["INFLUXDB_BUCKET"]

client = InfluxDBClient(url=os.environ["INFLUXDB_URL"], token=token)

point = (
    Point("performance")
    .tag("git", data["git"])
    .tag("type", data["type"])
    .field("total-app-time", data["kokkos-kernel-data"]["total-app-time"])
    .field("total-kernel-times", data["kokkos-kernel-data"]["total-kernel-times"])
    .field(
        "total-non-kernel-times", data["kokkos-kernel-data"]["total-non-kernel-times"]
    )
    .field("percent-in-kernels", data["kokkos-kernel-data"]["percent-in-kernels"])
    .field("unique-kernel-calls", data["kokkos-kernel-data"]["unique-kernel-calls"])
    .time(datetime.utcnow(), WritePrecision.NS)
)

write_api = client.write_api(write_options=SYNCHRONOUS)
write_api.write(bucket, org, point)
