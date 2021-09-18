#!/usr/bin/env python3

from gabriel_server.network_engine import server_runner
import logging
import argparse
import importlib

DEFAULT_PORT = 9099
DEFAULT_NUM_TOKENS = 2
INPUT_QUEUE_MAXSIZE = 60

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def create_adapter(openvino, cpu_only, force_torch, use_myriad):
    """Create the best adapter based on constraints passed as CLI arguments."""

    if use_myriad:
        openvino = True
        if cpu_only:
            raise Exception("Cannot run with both cpu-only and Myriad options")

    if force_torch and openvino:
        raise Exception("Cannot run with both Torch and OpenVINO")

    if not openvino:
        if importlib.util.find_spec("torch") is None:
            logger.info("Could not find Torch")
            openvino = True
        elif not cpu_only:
            import torch

            if torch.cuda.is_available():
                logger.info("Detected GPU / CUDA support")
                from torch_adapter import TorchAdapter

                return TorchAdapter(False, DEFAULT_STYLE)
            else:
                logger.info("Failed to detect GPU / CUDA support")

    if not force_torch:
        if importlib.util.find_spec("openvino") is None:
            logger.info("Could not find Openvino")
            if openvino:
                raise Exception("No suitable engine")
        else:
            if not cpu_only and not use_myriad:
                from openvino.inference_engine import IEPlugin

                try:
                    IEPlugin("GPU")
                    logger.info("Detected iGPU / clDNN support")
                except RuntimeError:
                    logger.info("Failed to detect iGPU / clDNN support")
                    cpu_only = True

            logger.info("Using OpenVINO")
            logger.info("CPU Only: %s", cpu_only)
            logger.info("Use Myriad: %s", use_myriad)
            from openvino_adapter import OpenvinoAdapter

            adapter = OpenvinoAdapter(cpu_only, DEFAULT_STYLE,
                                      use_myriad=use_myriad)
            return adapter

    logger.info("Using Torch with CPU")
    from torch_adapter import TorchAdapter

    return TorchAdapter(True, DEFAULT_STYLE)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-t", "--tokens", type=int, default=DEFAULT_NUM_TOKENS,
        help="number of tokens")

    parser.add_argument(
        "-p", "--port", type=int, default=DEFAULT_PORT, help="Set port number")
    args = parser.parse_args()

    server_runner.run(websocket_port=args.port, zmq_address='tcp://*:5555', num_tokens=args.tokens,
                  input_queue_maxsize=INPUT_QUEUE_MAXSIZE)


if __name__ == "__main__":
    main()
