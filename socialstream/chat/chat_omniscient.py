import json
import time
from typing import cast

import streamlit as st
from sotopia.database import (
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
    EpisodeLog,
)
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.parallel import (
    _agent_profile_to_friendabove_self,
    render_text_for_agent,
)
from sotopia.messages import AgentAction

from socialstream.utils import (
    HUMAN_MODEL_NAME,
    MODEL_LIST,
    ActionState,
    EnvAgentProfileCombo,
    initialize_session_state,
    messageForRendering,
    print_current_speaker,
    render_for_humans,
    set_from_env_agent_profile_combo,
    set_settings,
    step,
)


def chat_demo() -> None:
    initialize_session_state()

    def other_choice_callback() -> None:
        st.session_state.agent_models = [
            st.session_state.agent1_model_choice,
            st.session_state.agent2_model_choice,
        ]
        st.session_state.editable = st.session_state.edit_scenario
        print("Editable: ", st.session_state.editable)
        agent_choice_1 = st.session_state.agent_choice_1
        agent_choice_2 = st.session_state.agent_choice_2
        set_settings(
            agent_choice_1=agent_choice_1,
            agent_choice_2=agent_choice_2,
            scenario_choice=st.session_state.scenario_choice,
            user_agent_name="PLACEHOLDER",
            agent_names=[
                st.session_state.agent_choice_1,
                st.session_state.agent_choice_2,
            ],
            reset_agents=False,
        )

    def env_agent_choice_callback() -> None:
        if st.session_state.active:
            st.warning("Please stop the conversation first.")
            st.stop()

        agent_choice_1 = st.session_state.agent_choice_1
        agent_choice_2 = st.session_state.agent_choice_2
        if agent_choice_1 == agent_choice_2:
            st.warning(
                "The two agents cannot be the same. Please select different agents."
            )
            st.session_state.error = True
            return

        set_settings(
            agent_choice_1=agent_choice_1,
            agent_choice_2=agent_choice_2,
            scenario_choice=st.session_state.scenario_choice,
            user_agent_name="PLACEHOLDER",
            agent_names=[
                st.session_state.agent_choice_1,
                st.session_state.agent_choice_2,
            ],
            reset_agents=True,
        )

    st.checkbox(
        "Make the scenario editable",
        key="edit_scenario",
        on_change=other_choice_callback,
        disabled=st.session_state.active,
    )

    with st.expander("Create your scenario!", expanded=True):
        scenarios = st.session_state.env_mapping
        agent_list_1, agent_list_2 = st.session_state.agent_mapping

        scenario_col, scenario_desc_col = st.columns(2)
        with scenario_col:
            st.selectbox(
                "Choose a scenario:",
                scenarios.keys(),
                disabled=st.session_state.active,
                index=0,
                on_change=env_agent_choice_callback,
                key="scenario_choice",
            )

        with scenario_desc_col:
            st.markdown(
                f"""**Scenario Description:** {st.session_state.env_description_mapping[st.session_state.scenario_choice]}""",
                unsafe_allow_html=True,
            )

        agent_col1, agent_col2 = st.columns(2)
        with agent_col1:
            agent_choice_1 = st.selectbox(
                "Choose the first agent:",
                agent_list_1.keys(),
                disabled=st.session_state.active,
                index=0,
                on_change=env_agent_choice_callback,
                key="agent_choice_1",
            )
        with agent_col2:
            agent_choice_2 = st.selectbox(
                "Choose the second agent:",
                agent_list_2.keys(),
                disabled=st.session_state.active,
                index=1,
                on_change=env_agent_choice_callback,
                key="agent_choice_2",
            )
        if agent_choice_1 == agent_choice_2:
            st.warning(
                "The two agents cannot be the same. Please select different agents."
            )
            st.stop()

        model_col_1, model_col_2 = st.columns(2)
        with model_col_1:
            st.selectbox(
                "Choose a model:",
                MODEL_LIST,
                disabled=st.session_state.active,
                index=0,
                on_change=other_choice_callback,
                key="agent1_model_choice",
            )
        with model_col_2:
            st.selectbox(
                "Choose a model for agent 2:",
                MODEL_LIST,
                disabled=st.session_state.active,
                index=0,
                on_change=other_choice_callback,
                key="agent2_model_choice",
            )

    def edit_callback(reset_msgs: bool = False) -> None:
        env_profiles: EnvironmentProfile = st.session_state.env.profile
        env_profiles.scenario = st.session_state.edited_scenario
        agent_goals = [st.session_state[f"edited_goal_{i}"] for i in range(2)]
        env_profiles.agent_goals = agent_goals

        print("Edited scenario: ", env_profiles.scenario)
        print("Edited goals: ", env_profiles.agent_goals)

        env_agent_combo = EnvAgentProfileCombo(
            env=env_profiles,
            agents=[agent.profile for agent in st.session_state.agents.values()],
        )
        set_from_env_agent_profile_combo(
            env_agent_combo=env_agent_combo, reset_msgs=reset_msgs
        )

    with st.expander("Check your social task!", expanded=True):
        agent_infos = compose_agent_messages()
        env_info, goals_info = compose_env_messages()

        if st.session_state.editable:
            st.text_area(
                label="Change the scenario here:",
                value=f"""{env_info}""",
                height=150,
                on_change=edit_callback,
                key="edited_scenario",
                disabled=st.session_state.active or not st.session_state.editable,
            )

            agent1_col, agent2_col = st.columns(2)
            agent_cols = [agent1_col, agent2_col]
            for agent_idx, agent_info in enumerate(agent_infos):
                agent_col = agent_cols[agent_idx]
                with agent_col:
                    st.text_area(
                        label=f"Change the background info for Agent {agent_idx + 1} here:",
                        value=f"""{agent_info}""",
                        height=150,
                        disabled=st.session_state.active
                        or not st.session_state.editable,
                    )  # TODO not supported yet!!

            agent1_goal_col, agent2_goal_col = st.columns(2)
            agent_goal_cols = [agent1_goal_col, agent2_goal_col]
            for agent_idx, goal_info in enumerate(goals_info):
                agent_goal_col = agent_goal_cols[agent_idx]
                with agent_goal_col:
                    st.text_area(
                        label=f"Change the goal for Agent {agent_idx + 1} here:",
                        value=f"""{goal_info}""",
                        height=150,
                        key=f"edited_goal_{agent_idx}",
                        on_change=edit_callback,
                        disabled=st.session_state.active
                        or not st.session_state.editable,
                    )
        else:
            st.markdown(
                f"""**Scenario:** {env_info}""",
                unsafe_allow_html=True,
            )

            agent1_col, agent2_col = st.columns(2)
            agent_cols = [agent1_col, agent2_col]
            for agent_idx, agent_info in enumerate(agent_infos):
                agent_col = agent_cols[agent_idx]
                with agent_col:
                    st.markdown(
                        f"""**Agent {agent_idx + 1} Background:** {agent_info}""",
                        unsafe_allow_html=True,
                    )

            agent1_goal_col, agent2_goal_col = st.columns(2)
            agent_goal_cols = [agent1_goal_col, agent2_goal_col]
            for agent_idx, goal_info in enumerate(goals_info):
                agent_goal_col = agent_goal_cols[agent_idx]
                with agent_goal_col:
                    st.markdown(
                        f"""**Agent {agent_idx + 1} Goal:** {goal_info}""",
                    )

    def activate() -> None:
        st.session_state.active = True

    def activate_and_start() -> None:
        activate()

        env_agent_combo = EnvAgentProfileCombo(
            env=st.session_state.env.profile,
            agents=[agent.profile for agent in st.session_state.agents.values()],
        )
        set_from_env_agent_profile_combo(
            env_agent_combo=env_agent_combo, reset_msgs=True
        )

    action_taken: bool = False

    def stop_and_eval() -> None:
        if st.session_state != ActionState.IDLE:
            st.session_state.state = ActionState.EVALUATION_WAITING

    start_col, stop_col = st.columns(2)
    with start_col:
        start_button = st.button(
            "Start", disabled=st.session_state.active, on_click=activate_and_start
        )
        if start_button:
            # st.session_state.active = True
            st.session_state.state = ActionState.AGENT1_WAITING

    with stop_col:
        stop_button = st.button(
            "Stop", disabled=not st.session_state.active, on_click=stop_and_eval
        )
        if stop_button and st.session_state.active:
            st.session_state.state = ActionState.EVALUATION_WAITING
            with st.spinner("Evaluating..."):
                step(user_input="")
                action_taken = True

    requires_agent_input = (
        st.session_state.state == ActionState.AGENT1_WAITING
        and st.session_state.agent_models[0] == HUMAN_MODEL_NAME
    ) or (
        st.session_state.state == ActionState.AGENT2_WAITING
        and st.session_state.agent_models[1] == HUMAN_MODEL_NAME
    )

    requires_model_input = (
        st.session_state.state == ActionState.AGENT1_WAITING
        and st.session_state.agent_models[0] != HUMAN_MODEL_NAME
    ) or (
        st.session_state.state == ActionState.AGENT2_WAITING
        and st.session_state.agent_models[1] != HUMAN_MODEL_NAME
    )

    print_current_speaker()
    with st.form("user_input", clear_on_submit=True):
        user_input = st.text_input("Enter your message here:", key="user_input")

        if st.form_submit_button(
            "Submit",
            use_container_width=True,
            disabled=not requires_agent_input,
        ):
            with st.spinner("Agent acting..."):
                st.session_state.state = st.session_state.state + 1
                step(user_input=user_input)
                action_taken = True

    if requires_model_input:
        with st.spinner("Agent acting..."):
            st.session_state.state = st.session_state.state + 1
            step(user_input="")
            action_taken = True

    if st.session_state.state == ActionState.EVALUATION_WAITING:
        print("Evaluating...")
        with st.spinner("Evaluating..."):
            step()

    messages = render_messages()
    tag_for_eval = ["Agent 1", "Agent 2", "General"]
    chat_history = [
        message for message in messages if message["role"] not in tag_for_eval
    ]
    evaluation = [message for message in messages if message["role"] in tag_for_eval]

    with st.expander("Chat History", expanded=True):
        streamlit_rendering(chat_history)

    with st.expander("Evaluation"):
        # a small bug: when there is a agent not saying anything there will be no separate evaluation for that agent
        streamlit_rendering(evaluation)

    if action_taken:
        time.sleep(3)  # sleep for a while to prevent running too fast
        # TODO if the rerun is too fast then the message is not rendering
        st.rerun()


def streamlit_rendering(messages: list[messageForRendering]) -> None:
    agent1_name, agent2_name = list(st.session_state.agents.keys())[:2]
    agent_color_mapping = {
        agent1_name: "lightblue",
        agent2_name: "green",
    }

    avatar_mapping = {
        "env": "🌍",
        "obs": "🌍",
    }

    agent_names = [agent1_name, agent2_name]
    avatar_mapping = {
        agent_name: "👤"
        if st.session_state.agent_models[idx] == HUMAN_MODEL_NAME
        else "🤖"
        for idx, agent_name in enumerate(agent_names)
    }  # TODO maybe change the avatar because all bot/human will cause confusion

    role_mapping = {
        "Background Info": "background",
        "System": "info",
        "Environment": "env",
        "Observation": "obs",
        "General": "eval",
        "Agent 1": "eval",
        "Agent 2": "eval",
        agent1_name: agent1_name,
        agent2_name: agent2_name,
    }

    for index, message in enumerate(messages):
        role = role_mapping.get(message["role"], "info")
        content = message["content"]

        if role == "background":
            continue

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
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(content.replace("\n", "<br />"), unsafe_allow_html=True)


def compose_env_messages() -> tuple[str, list[str]]:
    env: ParallelSotopiaEnv = st.session_state.env
    env_profile = env.profile
    env_to_render = env_profile.scenario
    goals_to_render = env_profile.agent_goals

    return env_to_render, goals_to_render


def compose_agent_messages() -> list[str]:  # type: ignore
    agents = st.session_state.agents

    agent_to_render = [
        render_text_for_agent(
            raw_text=_agent_profile_to_friendabove_self(agent.profile, agent_id),
            agent_id=st.session_state.human_agent_idx,
        )
        for agent_id, agent in enumerate(agents.values())
    ]
    return agent_to_render


def render_messages() -> list[messageForRendering]:
    env = st.session_state.env
    agent_list = list(st.session_state.agents.values())

    epilog = EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        tag="tmp",
        models=[env.model_name, agent_list[0].model_name, agent_list[1].model_name],
        messages=[
            [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
            for messages_in_turn in st.session_state.messages
        ],
        reasoning=st.session_state.reasoning,
        rewards=st.session_state.rewards,
        rewards_prompt="",
    )
    rendered_messages = render_for_humans(epilog)
    return rendered_messages
