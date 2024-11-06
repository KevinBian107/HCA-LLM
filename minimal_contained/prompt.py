from LLM import *
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000

#TODO: Added Common Sense + Sensory Information

def judge_propmt_func(local_response, cen_response, prev_states):
    '''function for the judge to use'''

    #TODO: figure out can't have more explanations?
    judge_prompt = f'''
    You a a judger judgeing which agent in a grid-like field to move colored boxes is doing the correct move.
    You personally do not need to make any moves.

    The first agent is giving command of {cen_response}, but the second agent is sayin {local_response}.
    Here is all the previous actions of all the agents have taken : {prev_states}.
    Please judge which of the action from the first agent or the second agent is better.
    Do not come-up with something new, only choose one of them:

    Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}.
    Remanber to assign action to your self as well.
    Include an agent only if it has a task next.
    Now, plan the next step:
    '''
    return judge_prompt

def LLM_summarize_func(state_action_prompt_next_initial, model_name="qwen2.5:14b-instruct-q3_K_L"):
    '''Shorten the prompt given'''

    prompt1 = f"Please summarize the following content as concise as possible: \n{state_action_prompt_next_initial}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt1},
    ]
    response = LLaMA_response(messages, model_name)
    return response

def input_prompt_1_func(state_update_prompt):
    '''design input prompt'''

    user_prompt_1 = f"""
    You are a central planner directing agents in a grid-like field to move colored boxes.
    Each agent is assigned to a 1x1 square and can only interact with objects in its area.
    Agents can move a box to a neighboring square or a same-color target.
    Each square can contain many targets and boxes.
    The squares are identified by their center coordinates, e.g., square[0.5, 0.5].
    Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).
    Your task is to instruct each agent to match all boxes to their color-coded targets.
    After each move, agents provide updates for the next sequence of actions.
    Your job is to coordinate the agents optimally.
    {state_update_prompt}
    Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}.
    Include an agent only if it has a task next. Now, plan the next step:
    """
    return user_prompt_1

def rplh_prompt_func(state_update_prompt, data, dialogue_history_method):
    '''design input prompt for role-playing leader-hellucinating agent'''

    response_total_list = data['response_total_list']
    pg_state_list = data['pg_state_list']
    dialogue_history_list = data['dialogue_history_list']
    
    if len(pg_state_list) - len(response_total_list) != 1:
        raise ValueError("state and response list do not match")
    # if len(pg_state_list) - len(dialogue_history_list) != 1:
    #     raise ValueError("state and dialogue history list do not match")
    
    user_prompt_1 = f'''
    You are a central planner directing agents in a grid-like field to move colored boxes.
    Each agent is assigned to a 1x1 square and can only interact with objects in its area.
    Agents can move a box to a neighboring square or a same-color target.
    Each square can contain many targets and boxes.
    The squares are identified by their center coordinates, e.g., square[0.5, 0.5].
    Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).
    Your task is to instruct each agent to match all boxes to their color-coded targets.
    After each move, agents provide updates for the next sequence of actions.
    Your job is to coordinate the agents optimally.
    The previous state and action pairs at each step are:
    Please learn from previous steps.
    Not purely repeat the actions but learn why the state changes or remains in a dead loop.
    Avoid being stuck in action loops.
    Hence, the current state is {pg_state_list[-1]}, with the possible actions: {state_update_prompt}
    Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}.
    Include an agent only if it has a task next.
    Now, plan the next step:
    '''
    
    token_num_count = len(enc.encode(user_prompt_1))

    if dialogue_history_method in ("_w_only_state_action_history",
                                   "_w_compressed_dialogue_history",
                                   "_w_all_dialogue_history"):

        if dialogue_history_method == "_w_only_state_action_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        elif dialogue_history_method == "_w_compressed_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
                state_action_prompt_next = (
                    f"State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        elif dialogue_history_method == "_w_all_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break

        user_prompt_1 = f"""
        You are a central planner directing agent in a grid-like field to move colored boxes.
        You are agent at ...
        You need to make moves and other agents need to make moves as well.
        Each agent is assigned to a 1x1 square and can only interact with objects in its area.
        Agents can move a box to a neighboring square or a same-color target.
        Each square can contain many targets and boxes.
        The squares are identified by their center coordinates, e.g., square[0.5, 0.5].
        Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).
        Your task is to instruct each agent to match all boxes to their color-coded targets.
        After each move, agents provide updates for the next sequence of actions.
        Your job is to coordinate the agents optimally.
        The previous state and action pairs at each step are: {state_action_prompt}
        
        Please learn from previous steps.
        Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
        Hence, the current state is {pg_state_list[-1]}, with the possible actions: {state_update_prompt}
        Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}.
        Remanber to assign action to your self as well.
        Include an agent only if it has a task next.
        You do not need to say json format, just use it directly in the format of {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}.
        Now, plan the next step:
        """
    return user_prompt_1

def dialogue_func(
    state_update_prompt_local_agent,
    state_update_prompt_other_agent,
    central_response,
    data,
    dialogue_history_method):

    response_total_list = data['response_total_list']
    pg_state_list = data['pg_state_list']
    dialogue_history_list = data['dialogue_history_list']

    if len(pg_state_list) - len(response_total_list) != 1:
        raise ValueError("state and response list do not match")
    # if len(pg_state_list) - len(dialogue_history_list) != 1:
    #     raise ValueError("state and dialogue history list do not match")
    
    user_prompt_1 = f'''
    You're a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground.
    You can only interact with objects in your square.
    Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)).
    Each square can contain many targets and boxes.
    A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
    The previous state and action pairs at each step are:
    Please learn from previous steps.
    Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
    The central planner's current action plan is: {{{central_response}}}.
    Please evaluate the given plan. If you agree with it, respond 'I Agree', without any extra words.
    If not, briefly explain your objections to the central planner.
    Your response:
    '''
    token_num_count = len(enc.encode(user_prompt_1))

    if dialogue_history_method in (
        "_w_only_state_action_history",
        "_w_compressed_dialogue_history",
        "_w_all_dialogue_history",
    ):
        if dialogue_history_method == "_w_only_state_action_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        elif dialogue_history_method == "_w_compressed_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
                state_action_prompt_next = (
                    f"State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        elif dialogue_history_method == "_w_all_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break

        user_prompt_1 = f"""
        You are a central planner directing agent in a grid-like field to move colored boxes.
        Particularly, you're a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground.
        You can only interact with objects in your square.
        Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or nearby squares, represented by colors (e.g., move(box_red, target_red)).
        Each square can contain many targets and boxes.
        Other central planner is also coordinating all other agents to achieve the goal: match each box with its color-coded target.
        The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
        The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
        The previous state and action pairs at each step are: {state_action_prompt}
        Please learn from previous steps.
        Not purely repeat the actions but learn why the state changes or remains in a dead loop.
        Avoid being stuck in action loops.
        You can imagine first about how you would plan these actions and specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}.
        Remanber to assign action to your self as well.

        The other central planner's current action plan is giving as: {{{central_response}}}.
        Please be very critical in thinking about this plan.

        Please evaluate the given plan.
        If you agree with it, respond 'I Agree', without any extra words.
        If not, briefly explain your objections to this other central planner and an judger agent will get involved.
        Ensure that you still include the actions that you agree with. This is crucial, include all actions.
        Your response:
        """
    return user_prompt_1


def input_reprompt_func(state_update_prompt):
    user_reprompt = f"""
    Finished! The updated state is as follows(combined targets and boxes with the same color have been removed): {state_update_prompt}
    The output should be like json format like: {{Agent[0.5, 0.5]:move(box_blue, square[0.5, 1.5]), Agent[1.5, 0.5]:move...}}.
    If no action for one agent in the next step, just do not include its action in the output.
    Also remember at most one action for each agent in each step.
    Next step output:
    """
    return user_reprompt


def message_construct_func(
    user_prompt_list, response_total_list, dialogue_history_method
    ):

    messages = [{"role": "system", 
                 "content": f'''You are a helpful assistant. 
                 
                 When asked to specifiy your action plan, specificy it strictly in JSON format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_blue, target_blue])"}}. 
                 
                 Make sure that:
                 - If no action for an agent in the next step, do not include it in JSON output. 
                 - At most one action for each agent in each step.
                 '''
                 }]

    if f"{dialogue_history_method}" == "_w_all_dialogue_history":
        # print('length of user_prompt_list', len(user_prompt_list))
        for i in range(len(user_prompt_list)):
            messages.append({"role": "user", "content": user_prompt_list[i]})
            if i < len(user_prompt_list) - 1:
                messages.append(
                    {"role": "assistant", "content": response_total_list[i]}
                )
        # print('Length of messages', len(messages))
    elif f"{dialogue_history_method}" in (
        "_wo_any_dialogue_history",
        "_w_only_state_action_history",
    ):
        messages.append({"role": "user", "content": user_prompt_list[-1]})
        # print('Length of messages', len(messages))
    return messages