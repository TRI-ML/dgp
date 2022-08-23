# (Optional) Setup AWS for DGP

## Prerequisites

DGP provides utilities functions to download/upload data to
[AWS S3](https://aws.amazon.com/pm/serv-s3/). You need an AWS account created to
use DGP AWS dependent functions.

## Install AWS CLI

From the terminal, run:

```sh
aws --version
```

If you get a response, then you already have AWS CLI installed and can move on
to the AWS CLI Configuration section.

Otherwise, you need to install the AWS CLI.

```sh
sudo apt install awscli
aws --version
```

## AWS CLI Configuration

Once the AWS CLI is installed, configure the AWS credentials using your AWS
Access Key ID and AWS Secret Access Key, you can leave the default format to
None:

```sh
aws configure
```

Confirm the AWS crednentials is setup in environment variables via:

```sh
$ env | grep AWS
AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
AWS_DEFAULT_REGION=<YOUR_AWS_DEFAULT_REGION>
AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
```
