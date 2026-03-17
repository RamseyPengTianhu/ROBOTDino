# pip install websocket-client
import os
import json
import websocket

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"

PROMPT = """
You are RoboDino's intent parser.

Your job:
- Read the user's input
- Output ONLY valid JSON
- No explanation, no markdown

Allowed behaviors:
- greet
- curious
- excited
- present
- goodbye

Return exactly:
{"behavior":"<one_of_the_allowed_values>"}
"""

def send_json(ws, obj):
    ws.send(json.dumps(obj))

def extract_text_from_response_done(event: dict) -> str:
    # 尽量兼容不同输出结构，先把 text 拼出来
    texts = []
    response = event.get("response", {})
    for item in response.get("output", []):
        for part in item.get("content", []):
            if part.get("type") in ("output_text", "text"):
                if "text" in part:
                    texts.append(part["text"])
    return "".join(texts).strip()

def on_open(ws):
    print("Connected.")

    # 1) 配置 session
    send_json(ws, {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "output_modalities": ["text"],
            "instructions": PROMPT,
        }
    })

    # 2) 发一个测试用户输入
    send_json(ws, {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hello little dinosaur"}
            ]
        }
    })

    # 3) 触发模型生成
    send_json(ws, {
        "type": "response.create",
        "response": {
            "output_modalities": ["text"]
        }
    })

def on_message(ws, message):
    event = json.loads(message)
    etype = event.get("type")

    if etype == "session.created":
        print("session.created")

    elif etype == "session.updated":
        print("session.updated")

    elif etype == "response.output_text.delta":
        # 可选：流式打印
        delta = event.get("delta", "")
        print(delta, end="", flush=True)

    elif etype == "response.done":
        print("\nresponse.done")
        text = extract_text_from_response_done(event)
        print("Final text:", text)

        try:
            result = json.loads(text)
            behavior = result["behavior"]
            print("Behavior =", behavior)
            # 在这里接你的 Go2 planner
            # perform_behavior(behavior)
        except Exception as e:
            print("Failed to parse JSON:", e)
            print(json.dumps(event, indent=2))

    elif etype == "error":
        print("ERROR:", json.dumps(event, indent=2))

def on_error(ws, error):
    print("WS ERROR:", error)

def on_close(ws, code, msg):
    print("Closed:", code, msg)

ws = websocket.WebSocketApp(
    URL,
    header=[f"Authorization: Bearer {OPENAI_API_KEY}"],
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

ws.run_forever()