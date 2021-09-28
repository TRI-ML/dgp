(Optional) Setup AWS for DGP
============================

## Prerequisites

DGP provides utilities functions to download/upload data to [AWS S3](https://aws.amazon.com/pm/serv-s3/?trk=ps_a134p000004f2aOAAQ&trkCampaign=acq_paid_search_brand&sc_channel=PS&sc_campaign=acquisition_US&sc_publisher=Google&sc_category=Storage&sc_country=US&sc_geo=NAMER&sc_outcome=acq&sc_detail=aws%20s3&sc_content=S3_e&sc_matchtype=e&sc_segment=488982706722&sc_medium=ACQ-P|PS-GO|Brand|Desktop|SU|Storage|S3|US|EN|Text&s_kwcid=AL!4422!3!488982706722!e!!g!!aws%20s3&ef_id=CjwKCAjwndCKBhAkEiwAgSDKQf3JXhu5vTE9JDDoQpvQTUcRHNW2ugabpbhrGzns97aLMOvopVZMzRoCavEQAvD_BwE:G:s&s_kwcid=AL!4422!3!488982706722!e!!g!!aws%20s3). You need an AWS account created to use DGP AWS dependent functions.

## Install AWS CLI

From the terminal, run:

```sh
$ aws --version
```
If you get a response, then you already have AWS CLI installed and can move on to the AWS CLI Configuration section.

Otherwise, you need to install the AWS CLI.

```sh
$ sudo apt install awscli
$ aws --version
```

## AWS CLI Configuration
Once the AWS CLI is installed, configure the AWS credentials using your AWS Access Key ID and AWS Secret Access Key, you can leave the default format to None:

```sh
$ aws configure
```

Confirm the AWS crednentials is setup in environment variables via:

```sh
$ env | grep AWS
```

You should see the response:

```sh
AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
AWS_DEFAULT_REGION=<YOUR_AWS_DEFAULT_REGION>
AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
```
