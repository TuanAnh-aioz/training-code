import argparse
import logging
import os
import sys
import traceback

import aioz_ainode_adapter
import GPUtil
import psutil
from aioz_ainode_base.log import setup_logging

import aioz_trainer

logger = logging.getLogger(__name__)

VERSION = "2.2.0-training"

parser = argparse.ArgumentParser()
parser.add_argument("--version", "-v", action="store_true", default=False, help="package version")
parser.add_argument("--tmp_dir", type=str, default=None, help="temporary directory to store data")
parser.add_argument("--json_ip", type=str, default=None, help="path to input json file")
parser.add_argument("--estimate_resource", action="store_true", help="Return estimate resource in output")
args, unknown_args = parser.parse_known_args()


def get_resource() -> dict:
    resource = {"sys_ram": {"total": 0, "usage": 0}, "gpu_memory": {"total": 0, "usage": 0}}
    try:
        # GPU memory
        gpus = GPUtil.getGPUs()
        if len(gpus) > 0:
            gpu = gpus[0]
            resource["gpu_memory"]["total"] = gpu.memoryTotal
            resource["gpu_memory"]["usage"] = gpu.memoryUsed

        # System RAM
        process = psutil.Process(os.getpid())
        mem_total = round(int(psutil.virtual_memory().total) * 1e-6)  # Mb
        mem_usage = round(int(process.memory_full_info().rss) * 1e-6)
        resource["sys_ram"]["total"] = mem_total
        resource["sys_ram"]["usage"] = mem_usage
    except Exception as error:
        print(f"Failed to get system resources: {error}")
    return resource


def do_task(tmp_dir: str, json_ip: str, estimate_resource: bool):
    sys.path.insert(0, os.getcwd())

    input_object = aioz_ainode_adapter.adapter.dict_to_inputObj(json_pth=json_ip)
    setup_logging(log_file=os.path.join(input_object.working_directory, "log.log"), level="INFO")
    output_object = aioz_trainer.run(input_object)
    if estimate_resource:
        resource = get_resource()
        output_object = output_object.model_copy(update={"estimate_resource": resource})

    aioz_ainode_adapter.adapter.make_output(output_object, tmp_dir=tmp_dir)


def main():
    if args.version:
        print(f"version: {VERSION}")
        return

    if not os.getenv("ANODE_BASE_STARTED"):
        os.environ["ANODE_BASE_STARTED"] = "True"

        try:
            do_task(tmp_dir=args.tmp_dir, json_ip=args.json_ip, estimate_resource=args.estimate_resource)
        except Exception as error:
            aioz_ainode_adapter.adapter.make_error(msg=str(error), traceback=traceback.format_exc())


if __name__ == "__main__":
    main()
