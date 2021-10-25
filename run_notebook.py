#!/usr/bin/python
import src.utilities.environment as env
import subprocess
import argparse
import sys
import re
import os
import signal

from subprocess import PIPE, Popen
from threading import Thread

## Queue code from: https://stackoverflow.com/a/4896288
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # python 2.x

ON_POSIX = 'posix' in sys.builtin_module_names

DOCKER_CONTAINER_NAME = 'spmm-benchmark-notebook'

dlmc_path = env.find_dlmc()
print(f'Found DLMC dataset, using {dlmc_path}')

parser = argparse.ArgumentParser()
parser.add_argument('--build', help='build/re-build the docker container before running', action='store_true')
parser.add_argument('--log', help='write notebook url to ../running-notebook.txt', action='store_true')
args = parser.parse_args()

if args.build:
    subprocess.call(f'docker build {env.REPO_ROOT_PATH} -t {DOCKER_CONTAINER_NAME}', shell=True)


docker_args = [
    '--runtime nvidia',
    '--shm-size=32G',
    '-p 8888:8888 '
]

docker_mount_points = [
    f'{env.REPO_ROOT_PATH}:/mnt/spmm-benchmark',
    f'{dlmc_path}:/mnt/dlmc'
]

jupyter_ags = [
    '--NotebookApp.allow_origin=\'https://colab.research.google.com\'',
    '--port=8888',
    '--NotebookApp.port_retries=0',
    '--allow-root',
    '--ip 0.0.0.0'
]

docker_run_command_list = ['docker', 'run'] + docker_args + \
                          [f'-v {mnt}' for mnt in docker_mount_points] + \
                          [f'{DOCKER_CONTAINER_NAME}'] + \
                          ['/bin/bash', '-c'] + \
                          ['\"'] + \
                                'jupyter serverextension enable --py jupyter_http_over_ws'.split() + \
                                ['&&', 'jupyter', 'notebook'] + jupyter_ags + \
                          ['\"']


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


print('Running', " ".join(docker_run_command_list))
p = Popen(" ".join(docker_run_command_list),
          stdout=PIPE, stderr=PIPE, stdin=subprocess.PIPE, close_fds=ON_POSIX, shell=True)
q = Queue()
t = Thread(target=enqueue_output, args=(p.stderr, q))
t.daemon = True  # thread dies with the program
t.start()

def kill_docker():
    print('killing, ', p.pid)
    if p.pid is None:
        pass
    else:
        os.kill(p.pid, signal.SIGTERM)


import atexit
atexit.register(kill_docker)

# Block and read lines or wait for keyboard interrupt
found_token = False
notebook_started = False
while True:
    line = q.get().decode("utf-8")
    if not notebook_started:
        if 'Writing notebook server cookie secret' in line:
            notebook_started = True
            print('Notebook started')
        else:
            print(line, end='')
            continue

    line = q.get().decode("utf-8")

    if 'http' in line and not found_token:
        match = re.search(r'\?token=([0-9a-fA-F]+)', line)
        if match is not None:
            print(f'Token Found, use http://localhost:8888/?token={match.group(1)}')
            found_token = True

            with open(env.REPO_ROOT_PATH + '/../running-notebook.txt', 'w') as running_notebook_file:
                running_notebook_file.write(f'http://localhost:8888/?token={match.group(1)}\n')
