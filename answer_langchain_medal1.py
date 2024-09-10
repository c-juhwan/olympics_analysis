import json
import argparse
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_model(model_name, temperature=0, max_tokens=128):
    if "gpt" in model_name:
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    elif "claude" in model_name:
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens)
    elif "gemini" in model_name:
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    else:
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens,
                          openai_api_key="EMPTY",
                          openai_api_base="http://localhost:8000/v1")

def parse_medal_response(response_content, key):
    try:
        medal_response = int(response_content.split(f"{key}")[1].split("\n")[0].strip())
        return medal_response
    except:
        return -1 # not found

def get_initial_answer(args, data, model):
    for idx in tqdm(range(len(data["data"])), desc="Getting initial answer"):
        each_data = data["data"][idx]
        dialogue = each_data["dialogue"][:-2]  # Remove the last two messages

        if "gemma" in args.model_name:
            # Gemma does not support system messages
            # Prepend first system message to second human message
            dialogue[1][1] = dialogue[0][1] + " " + dialogue[1][1]
            # Remove first system message
            dialogue = dialogue[1:]

        try:
            response = model.invoke(dialogue)
        except Exception as e:
            print(dialogue)
            response = ""
            raise e

        # Parse the response
        response_content = response.content

        gold_medal_response = parse_medal_response(response_content, "Gold:")
        silver_medal_response = parse_medal_response(response_content, "Silver:")
        bronze_medal_response = parse_medal_response(response_content, "Bronze:")
        total_medal_response = parse_medal_response(response_content, "Total:")

        each_data["initial_answer"] = {
            "gold": gold_medal_response,
            "silver": silver_medal_response,
            "bronze": bronze_medal_response,
            "total": total_medal_response,
            "gold_correct": gold_medal_response == each_data["metadata"]["gold"],
            "silver_correct": silver_medal_response == each_data["metadata"]["silver"],
            "bronze_correct": bronze_medal_response == each_data["metadata"]["bronze"],
            "total_correct": total_medal_response == each_data["metadata"]["total"]
        }

        each_data["dialogue"][-2] = [
            each_data["dialogue"][-2][0],
            response_content
        ]

    # Calculate the accuracy
    gold_correct_count = sum(each_data["initial_answer"]["gold_correct"] for each_data in data["data"])
    silver_correct_count = sum(each_data["initial_answer"]["silver_correct"] for each_data in data["data"])
    bronze_correct_count = sum(each_data["initial_answer"]["bronze_correct"] for each_data in data["data"])
    total_correct_count = sum(each_data["initial_answer"]["total_correct"] for each_data in data["data"])
    total_data_count = len(data["data"])

    # make "result" key in data
    data["result"] = {
        "initial_answer_accuracy": {
            "gold": gold_correct_count / total_data_count,
            "silver": silver_correct_count / total_data_count,
            "bronze": bronze_correct_count / total_data_count,
            "total": total_correct_count / total_data_count
        }
    }

def get_final_answer(args, data, model):
    for idx in tqdm(range(len(data["data"])), desc="Getting final answer"):
        each_data = data["data"][idx]
        dialogue = each_data["dialogue"]  # Use the whole dialogue

        if "gemma" in args.model_name:
            # Gemma does not support system messages
            # Prepend first system message to second human message
            dialogue[1][1] = dialogue[0][1] + " " + dialogue[1][1]
            # Remove first system message
            dialogue = dialogue[1:]

        try:
            response = model.invoke(dialogue)
        except Exception as e:
            print(dialogue)
            response = ""
            raise e

        # Parse the response
        response_content = response.content

        if response_content.startswith("Yes"):
            gold_medal_response = each_data["initial_answer"]["gold"]
            silver_medal_response = each_data["initial_answer"]["silver"]
            bronze_medal_response = each_data["initial_answer"]["bronze"]
            total_medal_response = each_data["initial_answer"]["total"]
        else:
            gold_medal_response = parse_medal_response(response_content, "Gold:")
            silver_medal_response = parse_medal_response(response_content, "Silver:")
            bronze_medal_response = parse_medal_response(response_content, "Bronze:")
            total_medal_response = parse_medal_response(response_content, "Total:")

        each_data["final_answer"] = {
            "gold": gold_medal_response,
            "silver": silver_medal_response,
            "bronze": bronze_medal_response,
            "total": total_medal_response,
            "gold_correct": gold_medal_response == each_data["metadata"]["gold"],
            "silver_correct": silver_medal_response == each_data["metadata"]["silver"],
            "bronze_correct": bronze_medal_response == each_data["metadata"]["bronze"],
            "total_correct": total_medal_response == each_data["metadata"]["total"]
        }

        each_data["dialogue"].append([
            "ai",
            response_content
        ])

    # Calculate the accuracy
    gold_correct_count = sum(each_data["final_answer"]["gold_correct"] for each_data in data["data"])
    silver_correct_count = sum(each_data["final_answer"]["silver_correct"] for each_data in data["data"])
    bronze_correct_count = sum(each_data["final_answer"]["bronze_correct"] for each_data in data["data"])
    total_correct_count = sum(each_data["final_answer"]["total_correct"] for each_data in data["data"])
    total_data_count = len(data["data"])

    data["result"]["final_answer_accuracy"] = {
        "gold": gold_correct_count / total_data_count,
        "silver": silver_correct_count / total_data_count,
        "bronze": bronze_correct_count / total_data_count,
        "total": total_correct_count / total_data_count
    }

def save_data(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process medal data.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-08-06", help="Name of the model to use.")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature of the model.")
    parser.add_argument("--data_file", type=str, default="./data/question_medal1.json", help="Path to the data file.")
    parser.add_argument("--output_file", type=str, default="./result/answer_medal1_MODEL_NAME.json", help="Path to the output file.")
    args = parser.parse_args()  # Parse the arguments

    data = load_data(args.data_file)  # Use args.data_file
    model = get_model(args.model_name)  # Use args.model_name
    get_initial_answer(args, data, model)
    get_final_answer(args, data, model)

    # Change the order of the keys in the data: result, prompt, data
    data = {
        "result": {
            "model": args.model_name,
            "temperature": args.temperature,
            "initial_answer_accuracy": data["result"]["initial_answer_accuracy"],
            "final_answer_accuracy": data["result"]["final_answer_accuracy"]
        },
        "prompt": data["prompt"],
        "data": data["data"]
    }

    save_data(data, args.output_file)

if __name__ == "__main__":
    main()