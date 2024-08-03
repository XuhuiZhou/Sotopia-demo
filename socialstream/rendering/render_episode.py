import streamlit as st


def rendering_demo(messages) -> None:
    for index, message in enumerate(messages):
        content = message
        st.markdown(content.replace("\n", "<br />"), unsafe_allow_html=True)


def rendering(messages) -> None:
    # TODO make the rendering prettier
    pass
