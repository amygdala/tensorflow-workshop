import tensorflow as tf
import os
import logging
import sys
import ast

root = logging.getLogger()
root.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
root.addHandler(ch)

POD_NAME = os.environ.get('POD_NAME')
CLUSTER_CONFIG = os.environ.get('CLUSTER_CONFIG')
logging.info(POD_NAME)


def main(job_name, task_id, cluster_def):
    server = tf.train.Server(
        cluster_def,
        job_name=job_name,
        task_index=task_id
    )
    server.join()


if __name__ == '__main__':
    this_job_name, this_task_id, _ = POD_NAME.split('-')
    cluster_def = ast.literal_eval(CLUSTER_CONFIG)
    main(this_job_name, int(this_task_id), cluster_def)
