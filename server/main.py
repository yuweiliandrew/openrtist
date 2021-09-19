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
