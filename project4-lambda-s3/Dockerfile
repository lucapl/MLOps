# Start from the official AWS Lambda Python 3.8 image
# link:     https://gallery.ecr.aws/lambda/python
# FROM public.ecr.aws/lambda/python:3.8
FROM public.ecr.aws/lambda/python:3.13

# Install boto3 for interacting with AWS services and any other dependencies listed in requirements.txt (if present)
RUN pip install boto3

COPY ./lambda_src/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# note: assumes model.onnx in this folder:
COPY ./lambda_src/ ${LAMBDA_TASK_ROOT}

# Command to run the Lambda function when invoked by AWS Lambda
CMD ["lambda_function.lambda_handler"]
