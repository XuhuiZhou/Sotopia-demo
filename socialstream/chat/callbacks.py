import streamlit as st
from socialstream.utils import set_settings, set_from_env_agent_profile_combo, EnvAgentProfileCombo
from socialstream.rendering_utils import get_public_info, get_secret_info

from sotopia.database import AgentProfile, EnvironmentProfile

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
    
def agent_edit_callback(key: str = "") -> None:
    agents = list(st.session_state.agents.values())
    agents_info = [
        {
            "first_name": agent.profile.first_name,
            "last_name": agent.profile.last_name,
            "public_info": get_public_info(agent.profile, display_name=False),
            "secret": get_secret_info(agent.profile, display_name=False),
        }
        for agent in agents
    ]

    match key:
        case "edited_agent_0":
            agents_info[0]["public_info"] = st.session_state[key]
        case "edited_agent_1":
            agents_info[1]["public_info"] = st.session_state[key]
        case "edited_secret_0":
            agents_info[0]["secret"] = st.session_state[key]
        case "edited_secret_1":
            agents_info[1]["secret"] = st.session_state[key]

    agent_1_profile = AgentProfile(**agents_info[0])
    agent_2_profile = AgentProfile(**agents_info[1])
    print("Edited agent 1: ", agent_1_profile)
    print("Edited agent 2: ", agent_2_profile)

    env_agent_combo = EnvAgentProfileCombo(
        env=st.session_state.env.profile,
        agents=[agent_1_profile, agent_2_profile],
    )
    set_from_env_agent_profile_combo(
        env_agent_combo=env_agent_combo, reset_msgs=False
    )
    
def edit_callback(key: str = "", reset_msgs: bool = False) -> None:
    # set agent_goals and environment background
    env_profiles: EnvironmentProfile = st.session_state.env.profile
    scenario = env_profiles.scenario
    agent_goals = env_profiles.agent_goals

    match key:
        case "edited_scenario":
            scenario = st.session_state[key]
        case "edited_goal_0":
            agent_goals[0] = st.session_state[key]
        case "edited_goal_1":
            agent_goals[1] = st.session_state[key]
        # case _:
        #     raise ValueError(f"Invalid key: {key}")

    env_profiles.scenario = scenario
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

def agent_edit_callback_finegrained(key: str = "", traits: list[str] = []) -> None:
    agents = list(st.session_state.agents.values())
    agents_info = [
        {key: getattr(agent.profile, key) for key in traits} for agent in agents
    ]
    # "edited_agent_{agent_idx}_{trait_name}"
    trait = key.split("-")[-1]
    agent_idx = int(key.split("-")[-2])

    agents_info[agent_idx][trait] = st.session_state[key]
    agent_1_profile = AgentProfile(**agents_info[0])
    agent_2_profile = AgentProfile(**agents_info[1])
    print("Edited agent 1: ", agent_1_profile)
    print("Edited agent 2: ", agent_2_profile)

    env_agent_combo = EnvAgentProfileCombo(
        env=st.session_state.env.profile,
        agents=[agent_1_profile, agent_2_profile],
    )
    set_from_env_agent_profile_combo(
        env_agent_combo=env_agent_combo, reset_msgs=False
    )