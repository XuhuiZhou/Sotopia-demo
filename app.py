import os
import streamlit as st
os.environ["REDIS_OM_URL"] = st.secrets["REDIS_OM_URL"]
print(os.environ['REDIS_OM_URL'])
from sotopia.database import EpisodeLog

from haicosystem.utils.render import render_for_humans # type: ignore
from haicosystemDemo.hai_stream import streamlit_rendering
st.title("HAICosystem Episode Rendering")
# Text input for episode number
episode_number = st.text_input("Enter episode number:", value="2")
episode = EpisodeLog.find(EpisodeLog.tag == "haicosystem_debug")[int(episode_number)]  # type: ignore
assert isinstance(episode, EpisodeLog)
messages = render_for_humans(episode)
streamlit_rendering(messages)