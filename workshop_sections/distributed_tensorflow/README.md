
## Running Distributed Tensorflow in Kubernetes

[Kubernetes](http://kubernetes.io/) is Google's open source container orchestration framework. It provides a useful framework to run microservice based apps.

Tensorflow provides a distributed runtime with runs servers that communicate with [gRPC](http://www.grpc.io/) Google's high performance open-source RPC framework. Paired with Tensorflow's prepackaged docker containers, this makes Kubernetes a natural place to run distributed tensorflow.

### Create a Kubernetes Cluster on Google Container Engine

Follow the docs on [cloud.google.com](https://cloud.google.com/container-engine/docs/clusters/operations) to create a Kubernetes cluster. We recommend using the largest available machine type as this will increase performance. If you are on the free trial, the best cluster configuration available to you is likely (3 x `n1-highmem-2` nodes). We also recommend you install the [Google Cloud SDK](https://cloud.google.com/sdk/) and follow the `gcloud` tab in the cluster creation instructions, since we will use `gcloud` later in these instructions.

After you have created a cluster, you can use `gcloud` to authenticate to your Kubernetes master.

```sh
$ gcloud container clusters get-credentials <CLUSTER_NAME>
```

### Deploying a Tensorflow Cluster and Jupyter notebook server into your cluster

This repository comes prepackaged with [a template](templates/tensorflow-cluster.jinja) for distributed tensorflow clusters accessible through jupyter notebooks. This templating use a simplified version of the [Helm Chart](https://github.com/kubernetes/helm) template format, a package manager for Kubernetes.

We have also provided a [rendered.yaml](templates/rendered.yaml) which configures the provided cluster template with sensible defaults. To get started you simply need to create a [Kubernetes secret](http://kubernetes.io/docs/user-guide/secrets/) to serve as your jupyter password:

```
kubectl create secret generic jupyter --from-literal=password=[INSERT YOUR PASSWORD]
```

Then cd to the `templates` subdirectory, and deploy the rendered config to your cluster:

```sh
$ kubectl create -f rendered.yaml
```

Then, to get the IP for your Jupyter server, you can run:

```sh
$ kubectl get services jupyter-external
```

and use the listed `EXTERNAL_IP` to navigate to your Jupyter server. It may take a few minutes for the external IP address to appear.

A Tensorboard server is also available at `EXTERNAL_IP:6006`

### Running the example

Simply upload `intro_word2vec_destributed.ipynb` to your Jupyter server, and run through it.

### Cleanup

You can take down all your Kubernetes deployed pods and services by running:

```sh
$ kubectl create -f rendered.yaml
```

Then, if you want to take down your Kubernetes (Container Engine) cluster altogether, you can delete it from
the [Google Cloud Platform Console](https://console.cloud.google.com).
