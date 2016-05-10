import tensorflow as tf
import os
import logging
import argparse
import sys
import ast

root = logging.getLogger()
root.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
root.addHandler(ch)

NAMESPACE = os.environ.get('POD_NAMESPACE')
POD_NAME = os.environ.get('POD_NAME')

logging.info(POD_NAME)

JOB_NET_ADDR = '{job_name}-{task_id}.{namespace}.svc.cluster.local:{job_port}'


def job_config_to_cluster_def(job_config):
    return {
        job['name']: [
            JOB_NET_ADDR.format(
                job_name=job['name'],
                task_id=i,
                namespace=NAMESPACE,
                job_port=job.get('port', 8080)
            ) for i in range(0, job['num_tasks'])
        ] for job in job_config
    }


def main(job_config):
    this_job_name, this_task_id, _ = POD_NAME.split('-')
    cluster_def = job_config_to_cluster_def(job_config)
    server = tf.train.Server(
        cluster_def,
        job_name=this_job_name,
        task_index=int(this_task_id)
    )
    server.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-config',
        help='JSON String of a job config'
    )
    parsed = parser.parse_args()
    main(ast.literal_eval(parsed.job_config))
