import requests
from requests.auth import HTTPBasicAuth
import json
from urllib.parse import parse_qs
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from botocore.exceptions import ClientError
import boto3
import os
from langchain.prompts import PromptTemplate

# Initialize clients
s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")
bedrock_kb_client = boto3.client("bedrock-agent-runtime", region_name="us-west-2")

# Constants
DESTINATION_BUCKET = "cloudservices-chatbot"
CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
KNOWLEDGE_BASE_ID = "WN1KJW3LTV"
MODEL_ARN = "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
SLACK_VERIFICATION_TOKEN = os.environ["SLACK_VERIFICATION_TOKEN"]

# LangChain PromptTemplate
template = """
You are Claude, an AI assistant powered by Anthropic's Claude-3.5-Sonnet model, specialized in software development with access to a variety of tools and the ability to instruct and direct a coding agent and a code execution one
Given the user query and relevant knowledge base content, generate a detailed and helpful response.
Your capabilities include:

<capabilities>
1. Creating and managing project structures
2. Writing, debugging, and improving code across multiple languages
3. Providing architectural insights and applying design patterns
4. Staying current with the latest technologies and best practices
5. Analyzing and manipulating files within the project directory
6. Performing web searches for up-to-date information
7. Executing code and analyzing its output within an isolated 'code_execution_env' virtual environment
8. Managing and stopping running processes started within the 'code_execution_env'
9. Running shell commands.
</capabilities>

<error_handling>
Error Handling and Recovery:
- If a tool operation fails, carefully analyze the error message and attempt to resolve the issue.
- For file-related errors, double-check file paths and permissions before retrying.
- If a search fails, try rephrasing the query or breaking it into smaller, more specific searches.
- If code execution fails, analyze the error output and suggest potential fixes, considering the isolated nature of the environment.
- If a process fails to stop, consider potential reasons and suggest alternative approaches.
</error_handling>

<task_breakdown>
Break Down Complex Tasks:
When faced with a complex task or project, break it down into smaller, manageable steps. Provide a clear outline of the steps involved, potential challenges, and how to approach each part of the task.
</task_breakdown>

<explanation_preference>
Prefer Answering Without Code:
When explaining concepts or providing solutions, prioritize clear explanations and pseudocode over full code implementations. Only provide full code snippets when explicitly requested or when it's essential for understanding.
</explanation_preference>

<error_handling>
5. Error Handling:
   - If a tool operation fails, analyze the error and attempt to resolve the issue.
   - For persistent errors, consider alternative approaches to achieve the goal.
</error_handling>
User Query: {user_query}

Relevant Knowledge Base Content:
{kb_content}

Assistant:"""

prompt = PromptTemplate(input_variables=["user_query", "kb_content"], template=template)


def lambda_handler(event, context):
    # Debugging: Print the event object
    print(f"Received event: {json.dumps(event)}")

    # Step 1: Verify Slack request
    try:
        body = event["body"]
        params = parse_qs(body)
        token = params["token"][0]
        user_prompt = params.get("text", [""])[0]
        response_url = params["response_url"][0]

        if token != SLACK_VERIFICATION_TOKEN:
            raise ValueError("Invalid Slack verification token")

        print(f"Received prompt from Slack: {user_prompt}")
    except (KeyError, ValueError, IndexError) as e:
        print(f"Error parsing request body: {e}")
        return {
            "statusCode": 400,
            "body": "Invalid request body. Please provide a valid Slack event.",
        }

    # Step 2: Retrieve Knowledge Base data
    relevant_knowledge = ""
    try:
        objects = s3_client.list_objects_v2(Bucket=DESTINATION_BUCKET, Prefix="")
        if "Contents" in objects:
            for obj in objects["Contents"]:
                kb_key = obj["Key"]
                kb_response = s3_client.get_object(
                    Bucket=DESTINATION_BUCKET, Key=kb_key
                )
                try:
                    kb_content = kb_response["Body"].read().decode("utf-8")
                except UnicodeDecodeError as e:
                    print(f"Error decoding content from {kb_key}: {e}")
                    try:
                        kb_content = kb_response["Body"].read().decode("latin-1")
                    except UnicodeDecodeError as e:
                        print(f"Error decoding content from {kb_key} with latin-1: {e}")
                        continue
                if user_prompt.lower() in kb_content.lower():
                    relevant_knowledge += (
                        f"\nDocument: {kb_key}\nContent: {kb_content}\n"
                    )
                    print(f"Relevant content found in {kb_key}")
    except ClientError as e:
        print(f"Error accessing Knowledge Base: {e}")

    # Step 3: Retrieve additional knowledge from Bedrock
    try:
        response = bedrock_kb_client.retrieve_and_generate(
            input={"text": user_prompt},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN,
                },
            },
        )
        additional_kb_content = response["output"]["text"]
        relevant_knowledge += f"\nAdditional KB Content:\n{additional_kb_content}\n"
    except ClientError as e:
        print(f"Error invoking retrieve_and_generate: {e}")

    # Step 4: Construct the prompt using LangChain
    final_prompt = prompt.format(user_query=user_prompt, kb_content=relevant_knowledge)
    print(f"Constructed prompt using LangChain: {final_prompt}")

    # Step 5: Invoke Bedrock with LangChain-generated prompt
    bedrock_payload = {
        "modelId": CLAUDE_MODEL_ID,
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": 1,
            "top_p": 0.999,
            "messages": [{"role": "user", "content": final_prompt}],
        },
    }

    response_payload = ""
    try:
        print(f"Invoking model {CLAUDE_MODEL_ID} with Bedrock...")
        bedrock_response = bedrock_client.invoke_model_with_response_stream(
            modelId=bedrock_payload["modelId"],
            contentType=bedrock_payload["contentType"],
            accept=bedrock_payload["accept"],
            body=json.dumps(bedrock_payload["body"]),
        )

        for event in bedrock_response["body"]:
            if "chunk" in event:
                chunk_data = json.loads(event["chunk"]["bytes"].decode("utf-8"))
                if "delta" in chunk_data and "text" in chunk_data["delta"]:
                    response_payload += chunk_data["delta"]["text"]
    except ClientError as e:
        print(f"Error invoking Bedrock model: {e}")
        response_payload = "Error generating response."

    # Step 6: Send the response back to Slack
    slack_response = {
        "response_type": "in_channel",
        "text": f"*Prompt:* {user_prompt}\n*Response:* {response_payload}",
    }

    try:
        req = Request(
            response_url,
            method="POST",
            data=json.dumps(slack_response).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req) as res:
            res_body = res.read()
        print(f"Response sent to Slack: {res_body}")
    except (HTTPError, URLError) as e:
        print(f"Error sending response to Slack: {e}")
        return {"statusCode": 500, "body": "Error sending response to Slack"}

    return {"statusCode": 200, "body": json.dumps(slack_response)}
