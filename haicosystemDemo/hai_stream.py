import json
import streamlit as st
from haicosystem.protocols import messageForRendering # type: ignore

role_mapping = {
    "Background Info": "background",
    "System": "info",
    "Environment": "env",
    "Observation": "obs",
    "General": "info",
    "Agent 1": "info",
    "Agent 2": "info",
    "Xuhui Zhou": "human",
    "X AI": "ai",
}

avatar_mapping = {
    "env": "🌍",
    "obs": "🌍"
}

def streamlit_rendering(messages: list[messageForRendering]) -> None:
    for index, message in enumerate(messages):
        role = role_mapping.get(message["role"], "info")
        content = message["content"]
        
        if role == "obs" or message.get("type") == "action":
            content = json.loads(content)

        with st.chat_message(role, avatar=avatar_mapping.get(role, None)):
            if isinstance(content, dict):
                st.json(content)
            elif role == "info":
                st.markdown(
                    f"""
                    <div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
                        {content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif index < 2:  # Apply foldable for the first two messages
                st.markdown(
                    f"""
                    <details>
                        <summary>Message {index + 1}</summary>
                        {content}
                    </details>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(content.replace("\n", "<br />"), unsafe_allow_html=True)


