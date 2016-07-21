
## Running Distributed Tensorflow in Kubernetes

[Kubernetes](http://kubernetes.io/) is Google's open source container orchestration framework. It provides a useful framework to run microservice based apps.

Tensorflow provides a distributed runtime with runs servers that communicate with [gRPC](http://www.grpc.io/) Google's high performance open-source RPC framework. Paired with Tensorflow's prepackaged docker containers, this makes Kubernetes a natural place to run distributed tensorflow.

### Create a Kubernetes Cluster on Google Container Engine

Follow the docs on [cloud.google.com](https://cloud.google.com/container-engine/docs/clusters/operations) to create a Kubernetes cluster. We recommend using the largest available machine type as this will increase performance. If you are on the free trial, the best cluster configuration available to you is likely (3 x `n1-highmem-8` nodes). We also recommend you install the [Google Cloud SDK](https://cloud.google.com/sdk/) and follow the `gcloud` tab in the cluster creation instructions, since we will use `gcloud` later in these instructions.

After you have created a cluster, you can use `gcloud` to authenticate to your Kubernetes master

```
gcloud container clusters <CLUSTER_NAME> get-credentials
```

### Preparing your cluster

This repository comes prepackaged with a template for distributed tensorflow clusters and jupyter notebooks in `./k8s-configs/templates`. This templating use a simplified version of the [Helm Chart](https://github.com/kubernetes/helm) template format, a package manager for Kubernetes.

We have also provided a pre-rendered yaml files, which we will use for the workshop, but feel free to take a look at the templates, if you want to see how the configs were generated.

#### Create a NFS Server in your cluster

Tensorflow requires a distributed file system to checkpoint images. In this workshop we'll use a cluster local NFS server, but any file system that can be mounted read-write on multiple pods (RWX) in Kubernetes will work.

* First [create a GCE Persistent Disk](https://console.cloud.google.com/compute/disksAdd) to serve as the backend for your NFS server. Name it `nfs-backend`
* Create your NFS server with `kubectl create -f k8s-configs/nfs-server.yaml`
* Get the IP of your nfs server with `kubectl get services` and looks for `tf-nfs-server` and the `CLUSTER_IP` field
* Replace `0.0.0.0` in `k8s-configs/nfs-pvc.yaml` with the cluster IP of your NFS server
* Create the PersistentVolumeClaim for your NFS server with `kubectl create -f k8s-configs/nfs-pvc.yaml`

#### Load the workshop data onto your NFS Server

Use `kubectl create -f k8s-configs/load_data.yaml` to load the workshop data into your NFS server in the `/data` directory.

#### Create a Tensorboard Server

We'll start a Tensorboard Server in our cluster to display job logs. It will watch the `/output` directory on our NFS server for events written by TensorFlow

```
kubectl create -f k8s-configs/tensorboard.yaml
kubectl describe services tensorboard
```

Then to get to your Tensorboard server, navigate to the IP listed in the `LoadBalancer Ingress` field. Note it may take a few seconds to appear.

### Start a Tensorflow Job

To run the prebuilt job we use in this workshop you can simply run:

```
kubectl create -f k8s-configs/tf-cluster.yaml
```

Then navigate to your Tensorboard server to see the output! The source for this example can be found at in the [distributed_cnn](./distributed_cnn) directory.

If you want to run your own job follow the instructions below.

#### Build a Docker Image

Build a docker image to run your job `docker build -t gcr.io/<your-project-id>/tf-job`
Push your image to your Google Container registry for your project `docker push gcr.io/<your-project-id>/tf-job`

#### Create a Cluster Configuration using your image

Change the template file at `k8s-configs/templates/example-cluster.yaml` to use your image

NOTE: If you create a cluster configuration with a `:latest` tag you can use the same job configuration and simply update your docker image to rerun the job.

Then generate your cluster config from the template configuration (use a virtualenv to install python packages)

```
pip install -r k8s-configs/templates/requirements.txt
python k8s-configs/templates/render.py --config k8s-configs/templates/example-cluster.yaml --out ./my-cluster.yaml
kubectl create -f my-cluster.yaml
```

For more detailed explanation of distributed tensorflow, check out [the tensorflow docs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/how_tos/distributed/index.md).
