import os
from dotenv import load_dotenv
from mezo_agent import mezo_agent_musd_transaction, mezo_agent_transaction_btc
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool
from langsmith import Client
from pydantic import BaseModel, Field
from datetime import datetime
from types import SimpleNamespace

load_dotenv()

# Initialize LangSmith client (ensure your LANGSMITH_API_KEY is set if needed)
client = Client()

llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

musd_transaction_tool = Tool(
    name="musd_transaction_tool",
    func=mezo_agent_musd_transaction,
    description="Executes mUSD transfers on Mezo Matsnet. When a user instructs to send or transfer mUSD by specifying an amount and a recipient address, use this tool to process the transaction."
)

btc_transaction_tool = Tool(
    name="btc_transaction_tool",
    func=mezo_agent_transaction_btc,
    description="Executes BTC transfers on Mezo Matsnet. When a user instructs to send or transfer BTC by specifying an amount and a recipient address, use this tool to process the transaction."
)

tools = [musd_transaction_tool, btc_transaction_tool]
checkpointer = MemorySaver()

agent = create_react_agent(
    llm,
    tools,
    checkpointer=checkpointer
)

# ----------------------------------------------------------------
# LLM Judge Evaluator: Use the LLM to assess conceptual similarity.
# ----------------------------------------------------------------

judge_instructions = """Evaluate the student's answer against the ground truth for conceptual similarity regarding transaction type.
Accept the response if:
- For an mUSD request, the answer indicates that an mUSD transaction was executed successfully, including a transaction hash.
- For a BTC request, the answer indicates that a BTC transaction was executed successfully, including a transaction hash.
The transaction hash should be a 64-character hexadecimal string, but the exact value does not matter.
Return a JSON object with a boolean field "score" that is true if the response meets the appropriate criteria for the requested currency, and false otherwise.
Ensure the response is valid JSON and nothing else.
"""

class Grade(BaseModel):
    score: bool = Field(description="Indicates whether the response is conceptually accurate relative to the reference answer")

def evaluate_accuracy(outputs: dict, reference_outputs: dict) -> bool:
    prompt = f"Ground Truth answer: {reference_outputs['answer']};\nStudent's Answer: {outputs['response']}"
    messages = [
        {"role": "system", "content": judge_instructions},
        {"role": "user", "content": prompt}
    ]
    evaluation_result = llm.invoke(messages)
    evaluation_str = evaluation_result.content
    fixed_evaluation_str = evaluation_str.replace("True", "true").replace("False", "false")
    try:
        grade = Grade.model_validate_json(fixed_evaluation_str)
        return grade.score
    except Exception as e:
        print("Error parsing evaluation:", e)
        return False

# ----------------------------------------------------------------
# Create a dataset and upload examples programmatically.
# ----------------------------------------------------------------

# Expected output for every mUSD transfer evaluation.
expected_output_str = ("Transaction executed successfully. TX Hash: <tx_hash>")

dataset = client.create_dataset(
    dataset_name="btcvsmusd6",
    description="A sample dataset of mUSD transfer requests."
)

# Define examples as tuples: (input, expected_output)
examples = [
    # Ambiguous mUSD examples:
    ("Could you send some musd to 0x93a1Eadb069A791d23aAeDF3C272E2905bb63240?", expected_output_str),
    ("I want to transfer musd to 0x93a1Eadb069A791d23aAeDF3C272E2905bb63240, please process it.", expected_output_str),
    ("Initiate a musd payment to 0x93a1Eadb069A791d23aAeDF3C272E2905bb63240 if possible.", expected_output_str),
    ("Can you make a musd transaction to 0x93a1Eadb069A791d23aAeDF3C272E2905bb63240?", expected_output_str),
    ("Process a transfer using musd for me to 0x93a1Eadb069A791d23aAeDF3C272E2905bb63240.", expected_output_str),
    # Ambiguous BTC examples:
    ("Could you send a bit of btc to 0x93a1Eadb069A791d23aAeDF3C272E2905bb63240?", expected_output_str),
    ("I want to transfer bitcoin to 0x93a1Eadb069A791d23aAeDF3C272E2905bb63240.", expected_output_str),
    ("Initiate a btc payment to 0x93a1Eadb069A791d23aAeDF3C272E2905bb63240, please.", expected_output_str),
    ("Please process a bitcoin transaction to 0x93a1Eadb069A791d23aAeDF3C272E2905bb63240.", expected_output_str),
    ("Execute a transfer using btc to 0x93a1Eadb069A791d23aAeDF3C272E2905bb63240.", expected_output_str),
]

inputs = [{"input": inp} for inp, _ in examples]
outputs = [{"expected_output": out} for _, out in examples]

client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

# ----------------------------------------------------------------
# Define target function and evaluator for the experiment.
# ----------------------------------------------------------------

def target(inputs: dict) -> dict:
    input_text = inputs["input"]
    input_message = {"messages": [{"role": "user", "content": input_text}]}
    final_state = agent.invoke(input_message, config={"configurable": {"thread_id": 42}})
    response = final_state["messages"][-1].content
    return {"agent_output": response}

def accuracy(run, example, inputs, outputs, reference_outputs, attachments):
    eval_outputs = {"response": outputs["agent_output"]}
    expected = reference_outputs.get("expected_output") or reference_outputs.get("answer")
    eval_reference = {"answer": expected}
    return evaluate_accuracy(eval_outputs, eval_reference)

# ----------------------------------------------------------------
# Run the LangSmith evaluation experiment.
# ----------------------------------------------------------------

experiment_results = client.evaluate(
    target,
    data=dataset.id,
    evaluators=[accuracy],
    experiment_prefix="first-eval-in-langsmith",
    max_concurrency=2,
)

print("View evaluation results at:", experiment_results.results_url)

# ----------------------------------------------------------------
# Interactive REPL for live agent interactions.
# ----------------------------------------------------------------

while True:
    user_input = input("> ")
    if user_input.lower() in ["exit", "quit"]:
         break
    input_message = {"messages": [{"role": "user", "content": user_input}]}
    final_state = agent.invoke(input_message, config={"configurable": {"thread_id": 55}})
    response = final_state["messages"][-1].content
    print("Agent:", response)
    client.create_run(
         name="my-agent-run",
         run_type="chain",
         inputs={"user_input": user_input},
         outputs={"agent_output": response}
    )
