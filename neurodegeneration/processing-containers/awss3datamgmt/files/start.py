import os
from os import getenv
from os.path import join, exists, dirname, basename
from glob import glob
from pathlib import Path
import shutil
import uuid
import boto3
from botocore.exceptions import ClientError

# For multiprocessing -> usually you should scale via multiple containers!
from multiprocessing.pool import ThreadPool

# For shell-execution
from subprocess import PIPE, run

## For local testng
# os.environ["WORKFLOW_DIR"] = "/sharedFolder/EOR-230929143859926194" #"<your data directory>"
# os.environ["BATCH_NAME"] = "batch"
# os.environ["OPERATOR_IN_DIR"] = "archive"
# os.environ["OPERATOR_OUT_DIR"] = "output"
# os.environ["AWS_CREDENTIAL_FILE_PATH"] = "/sharedFolder/credentials"
# os.environ["AWS_CONFIG_FILE_PATH"] = str(None)
# os.environ["AWS_ACCESS_KEY"] = str(None)
# os.environ["AWS_SECRET_KEY"] = str(None)
# os.environ["S3_BUCKET_NAME"] = "data-coming-from-kaapana-develop"
# os.environ['S3_OBJECT_NAME_PREFIX']= 'S' #"dag-run-specific"
# os.environ["S3_ACTION"] = 'put'
# # os.environ["S3_OBJECT_SERIES_UID"] = "2.16.840.1.114362.1.12066432.24920037488.604832115.605.168"
# os.environ["S3_OBJECT_SERIES_DESCRIPTION"] = "T1"
# # os.environ["S3_OBJECT_GET_EXTENSION"]=".zip" #downloaded from S3 with this extension
# # os.environ["S3_OBJECT_PUT_SUFFIX"]="_0000_0000"
# # os.environ["S3_OBJECT_PUT_EXTENSION"]=".zip" #uploaded to s3 with this extension
# os.environ["INPUT_FILE_EXTENSION"]="*.zip"

execution_timeout = 300

# Counter to check if smth has been processed
processed_count = 0

aws_credential_file_path=os.environ["AWS_CREDENTIAL_FILE_PATH"]
aws_config_file_path=os.environ["AWS_CONFIG_FILE_PATH"]
aws_access_key=os.environ["AWS_ACCESS_KEY"]
aws_secret_key=os.environ["AWS_SECRET_KEY"]
s3_bucket_name=os.environ["S3_BUCKET_NAME"]
s3_object_name_prefix = os.environ['S3_OBJECT_NAME_PREFIX']
s3_action=os.environ["S3_ACTION"]
#seruid = os.environ["S3_OBJECT_SERIES_UID"]
series_description = os.environ["S3_OBJECT_SERIES_DESCRIPTION"]
s3_object_get_extension = os.environ["S3_OBJECT_GET_EXTENSION"]
s3_object_put_extension = os.environ["S3_OBJECT_PUT_EXTENSION"]
s3_object_put_suffix = os.environ["S3_OBJECT_PUT_SUFFIX"]#needed for older nnunet version
input_file_extension = os.environ["INPUT_FILE_EXTENSION"]

# set aws specific env variables if specified by user
if(aws_access_key != 'None'):
    os.environ['AWS_ACCESS_KEY_ID']=aws_access_key
    print("os.environ['AWS_ACCESS_KEY_ID']: ", os.environ['AWS_ACCESS_KEY_ID'])
if(aws_secret_key != 'None'):
    os.environ['AWS_SECRET_ACCESS_KEY']=aws_secret_key
    print("os.environ['AWS_SECRET_ACCESS_KEY']: ", os.environ['AWS_SECRET_ACCESS_KEY'])
if(aws_credential_file_path != 'None'):
    os.environ['AWS_SHARED_CREDENTIALS_FILE']=aws_credential_file_path
    print("os.environ['AWS_SHARED_CREDENTIALS_FILE']: ", os.environ['AWS_SHARED_CREDENTIALS_FILE'])
if(aws_config_file_path != 'None'):
    os.environ['AWS_CONFIG_FILE']=aws_config_file_path
    print("os.environ['AWS_CONFIG_FILE']: ", os.environ['AWS_CONFIG_FILE'])

#check valid action
if s3_action not in ['get', 'remove', 'put', 'list']:
    raise AssertionError('action must be get, remove, put or list')

if not os.path.exists('/tempIn'):
   os.makedirs('/tempIn')

if not os.path.exists('/tempOut'):
   os.makedirs('/tempOut')

def copy_file(src,dst):
    shutil.copyfile(src, dst)

def remove_files_from_folder(folder):
    folder += '/*.*'
    files = glob(folder, recursive=True)
    print(files)
    for f in files:
        os.remove(f)
        print('removing temp file: ', f)

def write_image(output, output_file_path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName ( str(output_file_path) )
    writer.Execute ( output )

def read_image(input_file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName ( str(input_file_path) )
    image = reader.Execute()
    return image

def get_seriesuid(bucket_name,object_key):
  s3 = boto3.client('s3')
  try:
    object_metadata = s3.head_object(Bucket=bucket_name, Key=object_key)['Metadata']  
  except ClientError as e:
        print(e)
        print(f'Error getting head object metadata- object: {object_key}, bucket: {bucket_name}.')
  seriesuid = object_metadata['seriesuid']
  print("series uid = ", seriesuid)

#retrieve objectname matching uid
def retrieve_objectname_matching_uid(bucket_name, seruid):
    print("#####Entered retrieve object name matching uid#####")
    s3 = boto3.client('s3')

    try:
        response = s3.list_objects(Bucket=bucket_name)  
    except ClientError as e:
        print(e)
        print(f'Error listing - bucket: {bucket_name}.')

    print("bucket name: ", bucket_name, "series uid: ", seruid)
    object_name = ""
    for content in response.get('Contents', []):
        object_key = content['Key']
        print('object_key: ', object_key)
        try:
            object_metadata = s3.head_object(Bucket=bucket_name, Key=object_key)['Metadata']
        except ClientError as e:
            print(e)
            print(f'Error getting head object metadata- object: {object_key}, bucket: {bucket_name}.')
        print("key: ", object_key, "object_metadata", object_metadata)
        if(object_metadata['seriesuid'] == seruid):
          print("found object with matchin series uid: ", object_key)
          object_name = object_key
          break
        else:
          print("no matching object found")
    return object_name
        
#list files in bucket
def list_files(bucket_name):
    print("####Entered list_files#####")
    print("bucket name: ", bucket_name)
    s3_client = boto3.client('s3')

    try:
        objects = s3_client.list_objects_v2(Bucket=bucket_name)
    except ClientError as e:
        print(e)
        print(f'Error listing -  bucket: {bucket_name}.')

    for obj in objects['Contents']:
        print(obj['Key'])
        get_seriesuid(bucket_name,obj['Key'])

    return objects

#upload file
def upload_file(file_name, bucket, uid,object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    print("#####Entered upload_file#####")
    print('object_name: ', object_name)
    print('file name: ', file_name)
    print("bucket name: ", bucket)
    print("series uid: ", uid)
    # If S3 object_name was not specified, use file_name
    if object_name == str(None):
        if(input_file_extension == "*.nii.gz"):
            object_name = os.path.basename(uid+'.nii.gz')
        else:
            object_name = os.path.basename(file_name)
        print('object_name: ', object_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name,ExtraArgs={'Metadata': {'seriesuid': uid,'seriesdescription':series_description}})
    except ClientError as e:
        print(e)
        print(f'Error uploading- file: {file_name}, object: {object_name}, bucket: {bucket_name}.')
        return False
    return True

def download_file(file_name,bucket_name,uid):
    print("######Entered download_file######")
    print('file name: ', file_name)
    print("bucket name: ", bucket_name)
    print("series uid: ", uid)
    s3 = boto3.client('s3')
    try:
        object_name = retrieve_objectname_matching_uid(bucket_name,uid)
        if(object_name == ""):
          print("object not found!")
        else:
          print("object name: ", object_name)
          response = s3.download_file(bucket_name, object_name, file_name)
    except ClientError as e:
        print(e)
        print(f'Error downloading- file: {file_name}, object: {object_name}, bucket: {bucket_name}.')
        return False
    return True

def remove_object(bucket_name, object_name):
    print("######Entered remove_object#######")
    print('object_name: ', object_name)
    print("bucket name: ", bucket_name)
    s3_client = boto3.client('s3')
    try:
        response = s3_client.delete_object(Bucket=bucket_name,Key=object_name)
    except ClientError as e:
        print(e)
        print(f'Error removing- object: {object_name}, bucket: {bucket_name}.')

def remove_all_objects(bucket_name):
    print("######Entered remove_all_objects########")
    print("bucket name: ", bucket_name)
    response = list_files(bucket_name)
    for object in response['Contents']:
        print('Deleting', object['Key'])
        remove_object(bucket_name, object['Key'])
        
def empty_bucket(bucket_name):
    print("######Entered empty_bucket#######")
    print("bucket name: ", bucket_name)
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(bucket_name).objects.delete()
    except s3.meta.client.exceptions as e:
        print(e)
        print(f'Error emptying - bucket: {bucket_name}.')
    
workflow_dir = getenv("WORKFLOW_DIR", "None")
workflow_dir = workflow_dir if workflow_dir.lower() != "none" else None
assert workflow_dir is not None

batch_name = getenv("BATCH_NAME", "None")
batch_name = batch_name if batch_name.lower() != "none" else None
assert batch_name is not None

operator_in_dir = getenv("OPERATOR_IN_DIR", "None")
operator_in_dir = operator_in_dir if operator_in_dir.lower() != "none" else None
assert operator_in_dir is not None

operator_out_dir = getenv("OPERATOR_OUT_DIR", "None")
operator_out_dir = operator_out_dir if operator_out_dir.lower() != "none" else None
assert operator_out_dir is not None

# File-extension to search for in the input-dir
# input_file_extension = "*.nii.gz"

# How many processes should be started?
parallel_processes = 1

print("##################################################")
print("#")
print("# Starting operator awsS3DataMgmt:")
print("#")
print(f"# workflow_dir:     {workflow_dir}")
print(f"# batch_name:       {batch_name}")
print(f"# operator_in_dir:  {operator_in_dir}")
print(f"# operator_out_dir: {operator_out_dir}")
print("#")
print("##################################################")
print("#")
print("# Starting processing on BATCH-ELEMENT-level ...")
print("#")
print("##################################################")
print("#")

# Loop for every batch-element (usually series)
batch_folders = sorted([f for f in glob(join("/", workflow_dir, batch_name, "*"))])
print('batch folders: ', batch_folders)
for batch_element_dir in batch_folders:
    print("batch element dir: ", batch_element_dir)
    print("#")
    print(f"# Processing batch-element {batch_element_dir}")
    print("#")
    element_input_dir = join(batch_element_dir, operator_in_dir)
    element_output_dir = join(batch_element_dir, operator_out_dir)

    # check if input dir present
    if not exists(element_input_dir):
        print("#")
        print(f"# Input-dir: {element_input_dir} does not exists!")
        print("# -> skipping")
        print("#")
        continue

    # creating output dir
    Path(element_output_dir).mkdir(parents=True, exist_ok=True)

    # creating output dir
    input_files = glob(join(element_input_dir, input_file_extension), recursive=True)
    print(f"# Found {len(input_files)} input-files!")

    # add a uuid to prefix to generate unique name
    # s3_object_name = s3_object_name_prefix +  uuid.uuid4().hex +  str(processed_count+1) + "_0000_0000.nii.gz"
    if(input_file_extension == "*.nii.gz"):
        s3_object_name = s3_object_name_prefix +  uuid.uuid4().hex +  str(processed_count+1) + s3_object_put_suffix + s3_object_put_extension
    else:
        s3_object_name = "None" #object name same as file name
    print('s3 object name: ', s3_object_name)
    # Single process:
    # Loop for every input-file found with extension 'input_file_extension'
    for input_file in input_files:

        if(input_file_extension == "*.nii.gz"):
            head, tail = os.path.split(input_file)
            # print('tail: ', tail)
            uid = os.path.basename(tail).split('.nii.gz', 1)[0]
        else:
            head2, tail2 = os.path.split(batch_element_dir)
            uid = tail2
        print('uid: ', uid)
        print(f'Applying action "{s3_action}" to files {input_file} in S3 bucket "{s3_bucket_name}"')
        if(s3_action == 'list'):
            list_files(s3_bucket_name)
            print("list files in s3 bucket done")
            processed_count += 1
        
        if(s3_action == 'put'):#upload file to S3
            upload_file(input_file,s3_bucket_name,uid,s3_object_name)
            print('upload to S3 bucket done.')
            processed_count += 1

        if(s3_action == 'get'):
            #output_file_path = os.path.join(element_output_dir, "{}.nii.gz".format(os.path.basename(batch_element_dir)))
            #it could either be .nii.gz or .zip ,the extension must be specified by user
            output_file_path = os.path.join(element_output_dir, "{}{}".format(os.path.basename(batch_element_dir),s3_object_get_extension))
            print("output_file_path: ",output_file_path)
            download_file(output_file_path,s3_bucket_name,uid)
            print("download from S3 bucket done.")
            processed_count += 1

        if(s3_action == 'remove'):
            if((s3_object_name_prefix == 'None') or (s3_object_name_prefix == '*.*')): #remove all objects from bucket
                #remove_all_objects(s3_bucket_name)
                empty_bucket(s3_bucket_name)
            elif(s3_object_name_prefix == 'dag-run-specific'):#remove object specific to the current run of dag
                object_name = retrieve_objectname_matching_uid(s3_bucket_name,uid)
                if(object_name == ""):
                    print("object not found!")
                else:
                    print("object name: ", object_name)
                    remove_object(s3_bucket_name,object_name)
            print("remove from S3 bucket done.")
            processed_count += 1


print("#")
print("##################################################")
print("#")
print("# BATCH-ELEMENT-level processing done.")
print("#")
print("##################################################")
print("#")

if processed_count == 0:
    print("#")
    print("##################################################")
    print("#")
    print("##################  ERROR  #######################")
    print("#")
    print("# ----> NO FILES HAVE BEEN PROCESSED!")
    print("#")
    print("##################################################")
    print("#")
    exit(1)
else:
    print("#")
    print(f"# ----> {processed_count} FILES HAVE BEEN PROCESSED!")
    print("#")
    print("# DONE #")

