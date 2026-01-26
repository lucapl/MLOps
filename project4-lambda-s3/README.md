# AWS lambda with S3 trigger

**IMPORTANT:** by default, docker pushes an "image index" for both arm and x86 cpu architectures.
Please build lambda Docker image with following options:

```
docker build  --provenance=false --platform linux/amd64 -t lambda-mlops:latest .
```

```
docker tag lambda-mlops:latest ....ecr.eu-central-1.amazonaws.com/mlops:latest
docker push ....ecr.eu-central-1.amazonaws.com/mlops:latest
```

## commands to interact with AWS S3

```
aws s3 ls s3://lambda-tutorial-mlops-1/ --human-readable --summarize

aws s3api put-object --bucket lambda-tutorial-mlops-1 --key digit2.png --body ./digit2.png

aws s3api get-object --bucket lambda-tutorial-mlops-1 --key digit2.png-preds.json   preds.json
```

## links

container image: https://gallery.ecr.aws/lambda/python

- Lambda functions tutorial: https://docs.aws.amazon.com/lambda/latest/dg/with-s3-example.html
- https://docs.aws.amazon.com/lambda/latest/dg/with-s3-tutorial.html
- https://github.com/aws-samples/aws-cdk-examples/tree/main/python/lambda-s3-trigger
