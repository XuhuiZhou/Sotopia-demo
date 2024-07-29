import os
import subprocess
import sys
import time
import streamlit as st
os.environ["REDIS_OM_URL"] = st.secrets["REDIS_OM_URL"]
print(os.environ['REDIS_OM_URL'])

from sotopia.database import EpisodeLog
from haicosystemDemo.hai_stream import streamlit_rendering

try:
  from haicosystem.utils.render import render_for_humans # type: ignore

except ModuleNotFoundError as e:
  subprocess.Popen([f'{sys.executable} -m pip install git+https://${st.secrets["GITHUB_TOKEN"]}@github.com/yourusername/yourrepo.git'], shell=True)
  # wait for subprocess to install package before running your actual code below
  time.sleep(90)

st.title("HAICosystem Episode Rendering")
# Text input for episode number
episode_number = st.text_input("Enter episode number:", value="2")
episode = EpisodeLog.find(EpisodeLog.tag == "haicosystem_debug")[int(episode_number)]  # type: ignore
assert isinstance(episode, EpisodeLog)
messages = render_for_humans(episode)
streamlit_rendering(messages)