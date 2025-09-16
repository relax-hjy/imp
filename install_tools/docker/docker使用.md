curl -I https://www.google.com

cat /etc/resolv.conf



export http_proxy="http://172.20.128.1:7890"
export https_proxy="http://172.20.128.1:7890"



#!/bin/bash
HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
export http_proxy="http://$HOST_IP:7890"
export https_proxy="http://$HOST_IP:7890"
echo "Proxy set to $HOST_IP:7890"





source proxy.sh