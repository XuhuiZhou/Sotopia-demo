import streamlit as st


def streamlit_rendering_basic(messages) -> None:
    for index, message in enumerate(messages):
        content = message
        st.markdown(content.replace("\n", "<br />"), unsafe_allow_html=True)


def streamlit_rendering(messages) -> None:
    # TODO make the rendering prettier
    pass
