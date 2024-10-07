from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from openai import OpenAI
import json

from dotenv import load_dotenv
load_dotenv()

from langsmith.wrappers import wrap_openai
from langsmith import traceable

client = wrap_openai(OpenAI())

@traceable
def prompt_compliance_evaluator(run: Run, example: Example) -> dict:
    # print("example:")
    # print(example.outputs)

    inputs = example.inputs['messages']
    outputs = example.outputs

    # print("Inputs:")
    # print(inputs)
    # print("Outputs:")
    # print(outputs)

    # Extract system prompt
    system_prompt = next((msg['data']['content'] for msg in inputs if msg['type'] == 'system'), "")

    # Extract message history
    message_history = []
    for msg in inputs:
        if msg['type'] in ['human', 'ai']:
            message_history.append({
                "role": "user" if msg['type'] == 'human' else "assistant",
                "content": msg['data']['content']
            })

    # Extract latest user message and model output
    latest_message = message_history[-1]['content'] if message_history else ""
    model_output = outputs['generations'][0]['text']

    evaluation_prompt = f"""
        System Prompt: {system_prompt}

        Message History:
        {json.dumps(message_history, indent=2)}

        Latest User Message: {latest_message}

        Model Output: {model_output}

        Based on the above information, evaluate the model's output for compliance with the system prompt and context of the conversation without loss of information. 
        Provide a score from 0 to 10, where 0 is completely non-compliant and 10 is perfectly compliant.
        Also provide a brief explanation for your score.

        Also, evaluate the model's output for how well organized and easy to follow the information is for a busy parent. 
        Provide a score from 0 to 10, where 0 is completely not organized or difficult to follow and 10 is perfectly organized and extremely easy to follow.
        Also provide a brief explanation for your score.

        Also, evaluate the model's output for how complete the information is. 0 for incomplete and 10 for very complete, with no relevant data missing.
        Also provide a brief explanation for your score.

        Respond in the following JSON format:
        {{
            "compliance": {{
                "score": <int>,
                "explanation": "<string>"
            }},
            "organization": {{
                "score": <int>,
                "explanation": "<string>"
            }}
            "completeness": {{
                "score": <int>,
                "explanation": "<string>"
            }}
        }}
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the compliance of model outputs to given prompts and conversation context."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.2
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "results": [{
                "key": "prompt_compliance",
                "score": result["compliance"]["score"] / 10,  # Normalize to 0-1 range
                "reason": result["compliance"]["explanation"]
            },
            {
                "key": "organization_score",
                "score": result["organization"]["score"] / 10,  # Normalize to 0-1 range
                "reason": result["organization"]["explanation"]
            },
            {
                "key": "completness_score",
                "score": result["completeness"]["score"] / 10,  # Normalize to 0-1 range
                "reason": result["completeness"]["explanation"]
            }],
        }
    except json.JSONDecodeError:
        return {
            "results": [{
                "key": "prompt_compliance",
                "score": 0,
                "reason": "Failed to parse evaluator response"
            },
            {
                "key": "organization_score",
                "score": 0,
                "reason": "Failed to parse evaluator response"
            },
            {
                "key": "completeness_score",
                "score": 0,
                "reason": "Failed to parse evaluator response"
            }
        ]}

# The name or UUID of the LangSmith dataset to evaluate on.
data = "Assistantly v2 Dataset"

# A string to prefix the experiment name with.
experiment_prefix = "Assistantly v2 prompt compliance"

# List of evaluators to score the outputs of target task
evaluators = [
    prompt_compliance_evaluator
]

# Evaluate the target task
results = evaluate(
    lambda inputs: inputs,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
)

print(results)
