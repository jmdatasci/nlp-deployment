#!/bin/sh

#SUBSCRIPTION_ID=6baae99a-4d64-4071-bfac-c363e71984c3
#az account set --subscription="${SUBSCRIPTION_ID}"

#minikube start --kubernetes-version=v1.22.6

az login --tenant berkeleydatasciw255.onmicrosoft.com
az account set --subscription=6baae99a-4d64-4071-bfac-c363e71984c3
az aks get-credentials --name w255-aks --resource-group w255 --overwrite-existing
kubectl config use-context w255-aks
kubelogin convert-kubeconfig

az acr login --name w255mids
az aks get-credentials --name w255-aks --resource-group w255 --overwrite-existing

IMAGE_PREFIX=jordanmeyer
IMAGE_NAME=project
ACR_DOMAIN=w255mids.azurecr.io
TAG=0964d58
IMAGE_FQDN="${ACR_DOMAIN}/${IMAGE_PREFIX}/${IMAGE_NAME}:${TAG}"
docker build --platform linux/amd64 --no-cache -t ${IMAGE_NAME}:${TAG} .
docker tag "${IMAGE_NAME}" ${IMAGE_FQDN}
docker push "${IMAGE_FQDN}"
docker pull "${IMAGE_FQDN}"

docker run --publish 8000:8000 --rm project:0964d58


#eval $(minikube -p minikube docker-env)
#docker build -t lab4 .
#kubectl kustomize .k8s/overlays/dev
#kubectl apply -k .k8s/overlays/dev
#minikube tunnel
#minikube ip
#kubectl port-forward service/lab4 8000:8000 &

NAMESPACE=jordanmeyer
kubectl --namespace externaldns logs -l "app.kubernetes.io/name=external-dns,app.kubernetes.io/instance=external-dns"
kubectl --namespace istio-ingress get certificates ${NAMESPACE}-cert
kubectl --namespace istio-ingress get certificaterequests
kubectl --namespace istio-ingress get gateways ${NAMESPACE}-gateway

kubectl config set-context --current --namespace=jordanmeyer

kubectl kustomize .k8s/bases
kubectl apply -k .k8s/bases
kubectl kustomize .k8s/overlays/dev
kubectl apply -k .k8s/overlays/dev


kubectl config use-context w255-aks
kubelogin convert-kubeconfig
kubectl kustomize .k8s/overlays/prod
kubectl apply -k .k8s/overlays/prod
kubectl config set-context --current --namespace=jordanmeyer

kubelogin convert-kubeconfig


cd ..
NAMESPACE=jordanmeyer

curl -X 'POST' 'https://jordanmeyer.mids-w255.com/predict' -L -H 'Content-Type: application/json' -d '{"text": ["I hate you.", "I love you."]}'

curl -X 'POST' 'http://localhost:8000/predict' -L -H 'Content-Type: application/json' -d '{"text": ["I hate you.", "I love you."]}'

kubectl logs -f $(kubectl get pods -A | grep project-7cff574ccd-s48pt | tail -1 | awk '{print $2}') --namespace=jordanmeyer

kubectl port-forward -n prometheus svc/grafana 3000:3000