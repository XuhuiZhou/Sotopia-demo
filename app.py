import os

import streamlit as st
from sotopia.database import EpisodeLog

from socialstream.chat import chat_demo
from socialstream.rendering import rendering_demo

print(os.environ["REDIS_OM_URL"])


DISPLAY_MODE = "Display Episodes"
CHAT_MODE = "Chat with Model"
st.set_page_config(page_title="SocialStream_Demo", page_icon="🧊", layout="wide")


def display() -> None:
    st.title("Episode Rendering...")
    # Text input for episode number
    episode_number = st.text_input("Enter episode number:", value="2")
    episode = EpisodeLog.find(EpisodeLog.tag == "gpt-4_gpt-4_v0.0.1_clean")[
        int(episode_number)
    ]  # type: ignore
    assert isinstance(episode, EpisodeLog)
    messages = episode.render_for_humans()[1]
    rendering_demo(messages)


option = st.sidebar.radio("Function", (CHAT_MODE, DISPLAY_MODE))
if option == DISPLAY_MODE:
    display()
elif option == CHAT_MODE:
    chat_demo()
