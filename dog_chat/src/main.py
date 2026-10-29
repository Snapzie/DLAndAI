from dotenv import load_dotenv
from uuid import uuid4
from agent import build_agent
from db import engine
load_dotenv()

def main():
    try:
        agent = build_agent()
        config = {"configurable": {"thread_id": str(uuid4())}}
        print("Ask a question! Type 'exit' to quit.")
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"exit", "quit"}:
                break

            if question.lower() == "debug":
                if result is not None:
                    for message in result["messages"]:
                        message.model_copy(update={"content": message.text}).pretty_print()
                else:
                    print("No conversation to inspect yet.")
                continue
            
            if not question:
                continue
            
            result = agent.invoke({"messages": [{"role": "user", "content": question}]},config=config)
            # print("Assistant:", result["messages"][-1].content)
            print("Assistant:", result["messages"][-1].text)
    finally:
        engine.dispose()

if __name__ == "__main__":
    main()