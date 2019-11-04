# Setup

## Development on AWS (in progress)

We will use the [Deep Learning Base AMI](https://aws.amazon.com/marketplace/pp/B07Y3VDBNS) which has NVIDA CUDA and GPU drivers, but no pre-installed deep learning framework Python packages (we will install those ourselves).

```sh
AMI="ami-0f4d5f31e6310624e"
TYPE="p2.4xlarge"
aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name id_rsa --security-group-ids=sg-331f3543
```

We'll tag it for later ease of reference

```sh
aws ec2 create-tags --resources <REPORTED InstanceId> --tags Key=Name,Value=fsdl
```

We also need to install aws CLI tools, and add two functions to our `.bashrc` or equivalent file

```sh
function ec2ip() {
    echo $(aws ec2 describe-instances --filters "{\"Name\":\"tag:Name\", \"Values\":[\"$1\"]}" --query='Reservations[0].Instances[0].PublicIpAddress' | tr -d '"')
}

function ec2id() {
    echo $(aws ec2 describe-instances --filters "{\"Name\":\"tag:Name\", \"Values\":[\"$1\"]}" --query='Reservations[0].Instances[0].InstanceId' | tr -d '"')
}
```
