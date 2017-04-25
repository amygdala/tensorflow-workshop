#!/bin/bash

gcloud ml-engine predict --model wnd1 --version vCMD2 --json-instances $1
