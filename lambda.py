import json
import boto3
import time
from botocore.exceptions import ClientError

# Initialize clients
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime', region_name='us-west-2')
bedrock_kb_client = boto3.client('bedrock-agent-runtime', region_name='us-west-2')

# Define your S3 bucket names and the correct model ID
SOURCE_BUCKET = 'cloudservices-chatbot'
DESTINATION_BUCKET = 'cloudservices-chatbot'
KNOWLEDGE_BASE_BUCKET = 'cloudservices-chatbot-kb'
KNOWLEDGE_BASE_PATH = ''
CLAUDE_MODEL_ID = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
KNOWLEDGE_BASE_ID = 'WN1KJW3LTV'
MODEL_ARN = 'arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0'

# Cache for Knowledge Base content to minimize S3 calls
knowledge_base_cache = None

def lambda_handler(event, context):
    jira_ticket_number = event['jira_ticket_number']
    source_key = f'source/CLOUD-{jira_ticket_number}.json'
    
    # Step 1: Retrieve Jira issue details from S3
    try:
        s3_response = s3_client.get_object(Bucket=SOURCE_BUCKET, Key=source_key)
        response_body = s3_response['Body'].read().decode('utf-8')
        print(f"Response from S3: {response_body}")
        issue_details = json.loads(response_body)
    except ClientError as e:
        print(f"Error fetching data from S3: {e}")
        return {"statusCode": 500, "body": f"Error fetching data from S3 for ticket {jira_ticket_number}"}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {"statusCode": 500, "body": "Error decoding the JSON data from S3. Please check the file format."}
    
    # Step 2: Retrieve Knowledge Base data (cached if previously retrieved)
    issue_description = issue_details.get('description', '')
    relevant_knowledge = ""
    global knowledge_base_cache
    if knowledge_base_cache is None:
        try:
            knowledge_base_cache = []
            objects = s3_client.list_objects_v2(Bucket=KNOWLEDGE_BASE_BUCKET, Prefix=KNOWLEDGE_BASE_PATH)
            if 'Contents' in objects:
                for obj in objects['Contents']:
                    kb_key = obj['Key']
                    kb_response = s3_client.get_object(Bucket=KNOWLEDGE_BASE_BUCKET, Key=kb_key)
                    try:
                        kb_content = kb_response['Body'].read().decode('utf-8')
                    except UnicodeDecodeError as e:
                        print(f"Error decoding content from {kb_key}: {e}")
                        continue
                    knowledge_base_cache.append((kb_key, kb_content))
        except ClientError as e:
            print(f"Error accessing Knowledge Base: {e}")
    
    # Filter cached KB content for relevant information
    for kb_key, kb_content in knowledge_base_cache:
        if issue_description.lower() in kb_content.lower():
            relevant_knowledge += f"\nDocument: {kb_key}\nContent: {kb_content}\n"
            print(f"Relevant content found in {kb_key}")
    
    # Step 3: Retrieve additional knowledge from Bedrock
    user_prompt = f"Here is an issue description: {issue_description}. Check for relevant solutions or rules."
    try:
        response = bedrock_kb_client.retrieve_and_generate(
            input={'text': user_prompt},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                    'modelArn': MODEL_ARN
                }
            }
        )
        
        additional_kb_content = response['output']['text']
        print(f"Additional content from Bedrock Knowledge Base: {additional_kb_content}")
        relevant_knowledge += f"\nAdditional KB Content:\n{additional_kb_content}\n"

    except ClientError as e:
        print(f"Error invoking retrieve_and_generate: {e}")
    
    # Step 4: Construct the prompt with combined knowledge
    if relevant_knowledge:
        prompt = f"\n\nHuman: Here is an issue description: {issue_description}\n\nRelevant Knowledge Base Content:\n{relevant_knowledge}\n\nAssistant:"
    else:
        prompt = f"\n\nHuman: {issue_description}\n\nAssistant:"

    bedrock_payload = {
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": 1,
            "top_p": 0.999,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
    }

    # Step 5: Retry Bedrock model invocation with exponential backoff
    max_retries = 4
    backoff_time = 2  # Start with 2 seconds
    response_payload = ""
    
    for attempt in range(max_retries):
        try:
            print(f"Invoking model {bedrock_payload['modelId']} with Bedrock (attempt {attempt + 1})...")
            bedrock_response = bedrock_client.invoke_model_with_response_stream(
                modelId=bedrock_payload["modelId"],
                contentType=bedrock_payload["contentType"],
                accept=bedrock_payload["accept"],
                body=json.dumps(bedrock_payload["body"])
            )

            for event in bedrock_response['body']:
                print(f"Event received from Bedrock: {event}")
                if 'chunk' in event:
                    chunk_data = json.loads(event['chunk']['bytes'].decode('utf-8'))
                    if 'delta' in chunk_data and 'text' in chunk_data['delta']:
                        response_payload += chunk_data['delta']['text']
            break  # Exit loop if successful

        except ClientError as e:
            print(f"Error invoking Bedrock model (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
            else:
                return {"statusCode": 500, "body": "Error processing data with Bedrock model"}
    
    if not response_payload:
        print("The response from Bedrock model is empty.")
        response_payload = "No response generated by the model."
    
    # Step 6: Write the response back to S3
    output_object_key = f'destination/response-{jira_ticket_number}.json'
    try:
        print(f"Writing response to S3 at {output_object_key}...")
        s3_client.put_object(
            Bucket=DESTINATION_BUCKET,
            Key=output_object_key,
            Body=json.dumps({'jira_ticket_number': jira_ticket_number, 'response': response_payload})
        )
    except ClientError as e:
        print(f"Error writing output to S3: {e}")
        return {"statusCode": 500, "body": f"Error writing output for ticket {jira_ticket_number} to S3"}
    
    return {"statusCode": 200, "body": f"Processed ticket {jira_ticket_number} successfully"}
