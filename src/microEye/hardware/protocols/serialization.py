from typing import Any

from microEye.hardware.protocols.actions import (
    BaseAction,
    ForLoop,
    FunctionCall,
    ParameterAdjustmentAction,
)
from microEye.hardware.protocols.actions_items import (
    ActionGroupItem,
    get_action_item,
)


def serialize_action(action: BaseAction) -> dict[str, Any]:
    '''
    Serialize an action to a JSON-serializable data structure.

    Parameters
    ----------
    action : BaseAction
        The action to be serialized.

    Returns
    -------
    dict[str, Any]
        A dictionary representing the serialized action data.
    '''
    action_data = {
        'type': type(action).__name__,
        'id': action.id,
        'name': action.name,
    }

    if isinstance(action, FunctionCall):
        action_data['function'] = action.function.__name__
        action_data['function_args'] = list(action.function_args)
    elif isinstance(action, ForLoop):
        action_data['repeat_count'] = action.repeat_count
        action_data['child_actions'] = [
            serialize_action(child) for child in action.child_actions
        ]
    elif isinstance(action, ParameterAdjustmentAction):
        action_data['target_object'] = action.target_object
        action_data['parameter_name'] = action.parameter_name
        action_data['parameter_value'] = action.parameter_value
        action_data['delay'] = action.delay
        action_data['is_expression'] = action.is_expression

    return action_data


def deserialize_action(action_data: dict[str, Any]) -> tuple[BaseAction, Any]:
    '''
    Deserialize an action from the JSON data structure.

    Parameters
    ----------
    action_data : dict[str, Any]
        The dictionary containing the serialized action data.

    Returns
    -------
    tuple[BaseAction, Any]
        A tuple containing the deserialized action instance and its corresponding
        action item.
    '''
    action_type = globals()[action_data['type']]

    if action_type == FunctionCall:
        function = globals()[action_data['function']]
        args = tuple(action_data['function_args'])
        action = FunctionCall(function, args)
        action_item = get_action_item(action)
    elif action_type == ForLoop:
        action = ForLoop(action_data['repeat_count'])
        action_item = ActionGroupItem(action)
        for child_data in action_data['child_actions']:
            child_action, child_item = deserialize_action(child_data)
            action.child_actions.append(child_action)
            action_item.add_child_item(child_item)
    elif action_type == ParameterAdjustmentAction:
        action = ParameterAdjustmentAction(
            target_object=action_data.get('target_object'),
            parameter_name=action_data.get('parameter_name'),
            parameter_value=action_data.get('parameter_value'),
            delay=action_data.get('delay', 0.0),
            is_expression=action_data.get('is_expression', False)
        )
        action_item = get_action_item(action)
    else:
        action = action_type()
        action_item = get_action_item(action)

    # action.id = action_data['id']
    action.name = action_data['name']

    return action, action_item
