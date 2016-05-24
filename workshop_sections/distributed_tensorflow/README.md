
## Running Distributed Tensorflow in Kubernetes

[Kubernetes](http://kubernetes.io/) is Google's open source container orchestration framework. It provides a useful framework to run microservice based apps.

Tensorflow provides a distributed runtime with runs servers that communicate with [gRPC](http://www.grpc.io/) Google's high performance open-source RPC framework. Paired with Tensorflow's prepackaged docker containers, this makes Kubernetes a natural place to run distributed tensorflow.

### Create a Kubernetes Cluster on Google Container Engine

Follow the docs on [cloud.google.com](https://cloud.google.com/container-engine/docs/clusters/operations) to create a Kubernetes cluster. We recommend using the largest available machine type as this will increase performance. If you are on the free trial, the best cluster configuration available to you is likely (3 x `n1-highmem-8` nodes). We also recommend you install the [Google Cloud SDK](https://cloud.google.com/sdk/) and follow the `gcloud` tab in the cluster creation instructions, since we will use `gcloud` later in these instructions.

After you have created a cluster, you can use `gcloud` to authenticate to your Kubernetes master

```
gcloud container clusters <CLUSTER_NAME> get-credentials
```

### Deploying a Tensorflow Cluster and Jupyter notebook server into your cluster

This repository comes prepackaged with a template for distributed tensorflow clusters and jupyter notebooks. This templating use a simplified version of the [Helm Chart](https://github.com/kubernetes/helm) template format, a package manager for Kubernetes.

To create the necessary objects in your cluster simply write a small config file to describe what you want your cluster to look like. Tensorflow divides its server instances into groups called `jobs` each of which have a number of identical `tasks`. So you must specify a number of jobs, each with a name and the number of tasks it should run. The jupyter server also requires you to specify a password.

You can check out the example config [here](templates/example-cluster.yaml). The only necessary modification will be your own `password` for the Jupyter server.

To render your config run (in your current conda environment)

```
conda install PyYaml jinja2
cd templates/
python render.py --config my-config.yaml --out rendered.yaml
```

Then to deploy your cluster

```
kubectl create -f rendered.yaml
```

Then, To get the IP for your Jupyter server, you can run

```
kubectl get services jupyter-external
```

and use the listed `EXTERNAL_IP` to navigate to your Jupyter server. A Tensorboard server is also available at `EXTERNAL_IP:6006`


For more detailed explanation of distributed tensorflow, check out [the tensorflow docs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/how_tos/distributed/index.md).

### Running the example

Simply upload `intro_word2vec_destributed.ipynb` to your Jupyter server, and run through it.
