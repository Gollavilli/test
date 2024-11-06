import json
import boto3
import time
from botocore.exceptions import ClientError

# Initialize clients
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime', region_name='us-west-2')
bedrock_kb_client = boto3.client('bedrock-agent-runtime', region_name='us-west-2')

# Define your S3 bucket names and the correct model ID
DESTINATION_BUCKET = 'cloudservices-chatbot'
CLAUDE_MODEL_ID = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
KNOWLEDGE_BASE_ID = 'WN1KJW3LTV'
MODEL_ARN = 'arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0'

# Cache for Knowledge Base content to minimize S3 calls
knowledge_base_cache = None

def lambda_handler(event, context):
    # Debugging: Print the event object
    print(f"Received event: {json.dumps(event)}")
    
    # Step 1: Retrieve the prompt from the API Gateway request
    try:
        if 'body' in event:
            body = json.loads(event['body'])
            user_prompt = body['text']
        elif 'prompt' in event:
            user_prompt = event['prompt']
        else:
            raise KeyError("Missing 'body' or 'prompt' in event")
        
        print(f"Received prompt: {user_prompt}")
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing request body: {e}")
        return {"statusCode": 400, "body": "Invalid request body. Please provide a valid JSON with a 'prompt' field."}
    
    # Step 2: Retrieve Knowledge Base data (cached if previously retrieved)
    relevant_knowledge = ""
    global knowledge_base_cache
    if knowledge_base_cache is None:
        try:
            knowledge_base_cache = []
            objects = s3_client.list_objects_v2(Bucket=DESTINATION_BUCKET, Prefix='')
            if 'Contents' in objects:
                for obj in objects['Contents']:
                    kb_key = obj['Key']
                    kb_response = s3_client.get_object(Bucket=DESTINATION_BUCKET, Key=kb_key)
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
        if user_prompt.lower() in kb_content.lower():
            relevant_knowledge += f"\nDocument: {kb_key}\nContent: {kb_content}\n"
            print(f"Relevant content found in {kb_key}")
    
    # Clear the cache to free up memory
    knowledge_base_cache = None
    
    # Step 3: Retrieve additional knowledge from Bedrock
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
        prompt = f"\n\nHuman: {user_prompt}\n\nRelevant Knowledge Base Content:\n{relevant_knowledge}\n\nAssistant:"
    else:
        prompt = f"\n\nHuman: {user_prompt}\n\nAssistant:"

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
    
    # Print the complete response
    print(f"Complete response from Bedrock model: {response_payload}")
    
    # Step 6: Write the response back to S3 (optional)
    output_object_key = f'destination/response-{user_prompt}.json'
    try:
        print(f"Writing response to S3 at {output_object_key}...")
        s3_client.put_object(
            Bucket=DESTINATION_BUCKET,
            Key=output_object_key,
            Body=json.dumps({'prompt': user_prompt, 'response': response_payload})
        )
    except ClientError as e:
        print(f"Error writing output to S3: {e}")
        return {"statusCode": 500, "body": f"Error writing output to S3"}
    
    # Return the response with both prompt and response
    return {
        "statusCode": 200,
        "body": json.dumps({
            "prompt": user_prompt,
            "response": response_payload
        }, indent=4)
    }
